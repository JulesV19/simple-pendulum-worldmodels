import torch
import torch.nn as nn
import torch.nn.functional as F

from ..encoder import ContextEncoder
from ..decoder import Decoder
from ..sigreg  import sigreg_loss
from ..losses  import PerceptualLoss, FrequencyLoss


class TransitionPredictor(nn.Module):
    """MLP de transition : z_t → ẑ_{t+1}. Identique à LeWorldModel."""

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, D) → ẑ: (B, T, D)"""
        return self.net(z)


class LeWorldModelRec(nn.Module):
    """
    LeWorldModel — reconstruction directe, sans JEPA.

    Pipeline (entraînement) :
      1. Encoder       : (frame_t, diff_t) → z_t                  (B, T, D)
      2. Rec loss      : MSE + perceptual(VGG) + freq(FFT)
                         sur decode(z_t) vs frame_t
      3. Pred loss k=1…rollout_k :
           z_roll = predictor^k(z_t)
           MSE + perceptual + freq sur decode(z_roll) vs frame_{t+k}
      4. SIGReg        : force z ~ N(0, I)

    Anti-collapse :
      Sans perceptual_coef > 0, l'encodeur s'effondre : la MSE pixel seule
      est minimisée par une image moyenne floue constante, ce qui annule le
      gradient vers l'encodeur. VGG16 est donc nécessaire, pas optionnel.
      freq_coef > 0 → FrequencyLoss (FFT) complémentaire, sans dépendances.

    Différence vs LeWorldModel :
      - Pas de target encoder EMA → pas de collapse silencieux possible
      - Supervision pixel directe → représentation plus interprétable
      - Le décodeur se partage entre rec et pred → cohérence espace latent

    Appeler model.update_target() est un no-op (API compat avec LeWorldModel).
    """

    def __init__(
        self,
        embed_dim:       int   = 128,
        hidden_dim:      int   = 512,
        lam:             float = 0.1,    # poids SIGReg
        rec_coef:        float = 1.0,    # poids reconstruction MSE
        pred_coef:       float = 1.0,    # poids prédiction rollout MSE
        perceptual_coef: float = 0.1,    # poids perceptual loss (VGG16) — 0 = désactivé
        freq_coef:       float = 0.05,   # poids frequency loss (FFT) — 0 = désactivé
        n_proj:          int   = 512,
        rollout_k:       int   = 5,
        # paramètres ignorés (compat CLI LeWorldModel)
        mse_coef:     float = 0.1,
        norm_coef:    float = 1.0,
        ema_momentum: float = 0.996,
        n_heads:      int   = 4,
        n_layers:     int   = 4,
        max_frames:   int   = 64,
        mask_ratio:   float = 0.0,
    ):
        super().__init__()
        self.embed_dim       = embed_dim
        self.lam             = lam
        self.rec_coef        = rec_coef
        self.pred_coef       = pred_coef
        self.perceptual_coef = perceptual_coef
        self.freq_coef       = freq_coef
        self.n_proj          = n_proj
        self.rollout_k       = rollout_k

        self.encoder   = ContextEncoder(embed_dim, in_channels=6)
        self.decoder   = Decoder(embed_dim)
        self.predictor = TransitionPredictor(embed_dim, hidden_dim)

        self.perceptual = PerceptualLoss() if perceptual_coef > 0 else None
        self.freq       = FrequencyLoss()  if freq_coef > 0       else None

    # ── Frame stacking ───────────────────────────────────────────────────────

    @staticmethod
    def _make_pairs(frames: torch.Tensor) -> torch.Tensor:
        """(B, T, 3, H, W) → (B, T, 6, H, W) : concat(frame_t, frame_t - frame_{t-1})"""
        diff = torch.zeros_like(frames)
        diff[:, 1:] = frames[:, 1:] - frames[:, :-1]
        return torch.cat([frames, diff], dim=2)

    # ── Forward (entraînement) ───────────────────────────────────────────────

    def forward(self, frames: torch.Tensor) -> dict:
        """
        Args:
            frames : (B, T, 3, H, W)  séquences normalisées [0, 1]

        Returns:
            dict : loss, rec_loss, pred_loss, perc_loss, freq_loss, sigreg (scalaires)
        """
        B, T, C, H, W = frames.shape
        pairs       = self._make_pairs(frames)
        frames_flat = pairs.reshape(B * T, 6, H, W)

        z = self.encoder(frames_flat).view(B, T, self.embed_dim)  # (B, T, D)

        # ── Reconstruction ────────────────────────────────────────────────────
        frames_hat = self.decoder(z)                               # (B, T, 3, H, W)
        rec_loss   = F.mse_loss(frames_hat, frames)

        perc_loss = (
            self.perceptual(frames_hat, frames) if self.perceptual is not None
            else torch.zeros(1, device=frames.device)
        )
        freq_loss = (
            self.freq(frames_hat, frames) if self.freq is not None
            else torch.zeros(1, device=frames.device)
        )

        # ── Prédiction multi-step en pixel space ──────────────────────────────
        pred_loss      = torch.zeros(1, device=frames.device)
        pred_perc_loss = torch.zeros(1, device=frames.device)
        pred_freq_loss = torch.zeros(1, device=frames.device)

        for k in range(1, self.rollout_k + 1):
            T_k           = T - k
            z_roll        = z[:, :T_k]                             # (B, T_k, D)
            for _ in range(k):
                z_roll    = self.predictor(z_roll)                 # (B, T_k, D)
            frames_pred   = self.decoder(z_roll)                   # (B, T_k, 3, H, W)
            frames_target = frames[:, k:k + T_k]                   # (B, T_k, 3, H, W)

            pred_loss = pred_loss + F.mse_loss(frames_pred, frames_target)
            if self.perceptual is not None:
                pred_perc_loss = pred_perc_loss + self.perceptual(frames_pred, frames_target)
            if self.freq is not None:
                pred_freq_loss = pred_freq_loss + self.freq(frames_pred, frames_target)

        pred_loss      = pred_loss      / self.rollout_k
        pred_perc_loss = pred_perc_loss / self.rollout_k
        pred_freq_loss = pred_freq_loss / self.rollout_k

        # ── SIGReg ────────────────────────────────────────────────────────────
        z_flat = z.reshape(B * T, self.embed_dim)
        sigreg = sigreg_loss(z_flat, self.n_proj)

        total_perc = perc_loss + pred_perc_loss
        total_freq = freq_loss + pred_freq_loss

        loss = (
            self.rec_coef        * rec_loss
            + self.pred_coef     * pred_loss
            + self.perceptual_coef * total_perc
            + self.freq_coef     * total_freq
            + self.lam           * sigreg
        )

        return {
            "loss":      loss,
            "rec_loss":  rec_loss.detach(),
            "pred_loss": pred_loss.detach(),
            "perc_loss": total_perc.detach(),
            "freq_loss": total_freq.detach(),
            "sigreg":    sigreg.detach(),
        }

    def update_target(self) -> None:
        """No-op — conservé pour compatibilité API avec LeWorldModel."""
        pass

    # ── Inférence ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """frames: (B, T, 3, H, W) → z: (B, T, embed_dim)"""
        B, T, C, H, W = frames.shape
        pairs = self._make_pairs(frames)
        z = self.encoder(pairs.reshape(B * T, 6, H, W))
        return z.view(B, T, self.embed_dim)

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, T, D) ou (B, D) → frames: (B, T, 3, 64, 64) ou (B, 3, 64, 64)"""
        return self.decoder(z)

    @torch.no_grad()
    def imagine(self, z0: torch.Tensor, n_steps: int) -> torch.Tensor:
        """
        Rollout depuis z0 en appliquant le MLP de transition pas à pas.

        Args:
            z0      : (B, embed_dim) ou (B, 1, embed_dim)
            n_steps : nombre de pas à prédire

        Returns:
            z_traj : (B, n_steps + 1, embed_dim)  — inclut z0
        """
        if z0.dim() == 2:
            z0 = z0.unsqueeze(1)
        traj = [z0]
        for _ in range(n_steps):
            traj.append(self.predictor(traj[-1]))
        return torch.cat(traj, dim=1)
