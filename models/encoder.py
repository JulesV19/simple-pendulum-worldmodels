import copy
import torch
import torch.nn as nn


class ContextEncoder(nn.Module):
    """
    CNN déterministe : (B, in_channels, 64, 64) → embedding ∈ R^embed_dim.

    in_channels=6 pour le frame stacking : concat(frame_t, frame_t - frame_{t-1}).
    Pas de mu/log_var, pas de reparameterize — l'espace latent n'est
    pas contraint par un prior gaussien. Le gradient passe librement.
    """

    def __init__(self, embed_dim: int = 256, in_channels: int = 6):
        super().__init__()
        self.embed_dim = embed_dim

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32,  4, stride=2, padding=1),  # → (32, 32, 32)
            nn.ReLU(),
            nn.Conv2d(32,  64,  4, stride=2, padding=1),  # → (64, 16, 16)
            nn.ReLU(),
            nn.Conv2d(64,  128, 4, stride=2, padding=1),  # → (128, 8, 8)
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # → (256, 4, 4)
            nn.ReLU(),
        )
        self.fc = nn.Linear(256 * 4 * 4, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_channels, 64, 64) ou (B*T, in_channels, 64, 64) → (B, embed_dim)"""
        return self.fc(self.conv(x).flatten(1))


class TargetEncoder(nn.Module):
    """
    Copie EMA du ContextEncoder — aucun gradient ne traverse ce module.

    Le target encoder fournit les embeddings "cibles" stables que le
    predictor doit apprendre à reproduire. Sa mise à jour lente (momentum)
    évite l'effondrement représentationnel sans nécessiter de contrastive loss.
    """

    def __init__(self, context_encoder: ContextEncoder, momentum: float = 0.996):
        super().__init__()
        self.momentum = momentum
        self.encoder  = copy.deepcopy(context_encoder)
        for p in self.encoder.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, context_encoder: ContextEncoder) -> None:
        """Mise à jour EMA — appeler après chaque optimizer.step()."""
        for p_ema, p_ctx in zip(self.encoder.parameters(),
                                context_encoder.parameters()):
            p_ema.data.mul_(self.momentum).add_(p_ctx.data,
                                                alpha=1.0 - self.momentum)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
