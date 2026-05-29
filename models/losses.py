import torch
import torch.nn as nn
import torch.nn.functional as F


def _subsample(pred: torch.Tensor, target: torch.Tensor, max_samples: int):
    """
    Flatten (..., C, H, W) → (N, C, H, W) et sous-échantillonne à max_samples.
    Le gradient reste actif sur pred, target est détaché.
    """
    C, H, W = pred.shape[-3], pred.shape[-2], pred.shape[-1]
    p = pred.reshape(-1, C, H, W)
    t = target.reshape(-1, C, H, W).detach()
    n = p.shape[0]
    if n > max_samples:
        idx = torch.randperm(n, device=p.device)[:max_samples]
        p = p[idx]
        t = t[idx]
    return p, t


class PerceptualLoss(nn.Module):
    """
    Perceptual loss basée sur VGG16 (Johnson et al., 2016).

    Compare les feature maps à trois profondeurs :
      relu1_2 → bords, couleurs        (64×64)
      relu2_2 → textures               (32×32)
      relu3_3 → structures             (16×16)

    Pourquoi ça évite le flou MSE :
      MSE minimise E[(p-t)²] → solution optimale = moyenne des modes
      → flou. VGG compare dans un espace où les activations sont éparses
      et non-linéaires → la "moyenne" n'est plus une solution facile.

    max_samples : nombre max de frames passées dans VGG par appel.
      Avec B=32, T=64 on aurait 2048 frames → OOM. On sous-échantillonne
      aléatoirement à max_samples avant chaque forward VGG.

    Input : tenseurs [0, 1] RGB. Normalisation ImageNet appliquée en interne.
    Handles (B, 3, H, W) et (B, T, 3, H, W).
    """

    # Indices dans vgg16.features (voir architecture VGG16)
    _SLICES = [
        (0,  4),   # relu1_2
        (4,  9),   # relu2_2  (MaxPool + block2)
        (9, 16),   # relu3_3  (MaxPool + block3)
    ]

    def __init__(
        self,
        weights:     tuple[float, ...] = (1.0, 1.0, 1.0),
        max_samples: int               = 64,
    ):
        super().__init__()
        self.max_samples = max_samples

        from torchvision import models
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        self.slices = nn.ModuleList([
            nn.Sequential(*list(vgg.features.children())[a:b])
            for a, b in self._SLICES
        ])
        self.weights = weights

        for p in self.parameters():
            p.requires_grad_(False)

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target : (..., 3, H, W) ∈ [0, 1]"""
        pred, target = _subsample(pred, target, self.max_samples)

        xp = (pred   - self.mean) / self.std
        xt = (target - self.mean) / self.std

        loss = torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
        for w, slice_ in zip(self.weights, self.slices):
            xp = slice_(xp)
            xt = slice_(xt)
            loss = loss + w * F.mse_loss(xp, xt)

        return loss


class FrequencyLoss(nn.Module):
    """
    Perte dans le domaine fréquentiel (FFT 2D).

    Pénalise les différences de spectre amplitude et phase entre pred et target.
    Complémentaire à MSE : MSE est aveugle aux hautes fréquences (textures,
    contours fins) car leur contribution à l'erreur L2 est faible en amplitude
    mais perceptuellement importante.

    max_samples : même logique que PerceptualLoss — la FFT est moins gourmande
      que VGG mais avec 2048 frames elle consomme quand même plusieurs GB.

    Pas de dépendances externes — utilise torch.fft.

    high_freq_boost : multiplie le spectre par une rampe fréquentielle pour
      amplifier les hautes fréquences → force encore plus de netteté.
    """

    def __init__(self, high_freq_boost: bool = True, max_samples: int = 256):
        super().__init__()
        self.high_freq_boost = high_freq_boost
        self.max_samples     = max_samples

    @staticmethod
    def _freq_weight(h: int, w: int, device: torch.device) -> torch.Tensor:
        """Rampe radiale : poids proportionnel à la fréquence spatiale."""
        fy = torch.fft.fftfreq(h, device=device).abs()
        fx = torch.fft.rfftfreq(w, device=device).abs()
        return (fy[:, None] + fx[None, :]).clamp(min=1e-6)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """pred, target : (..., 3, H, W) ∈ [0, 1]"""
        pred, target = _subsample(pred, target, self.max_samples)

        pred_f   = torch.fft.rfft2(pred,   norm="ortho")
        target_f = torch.fft.rfft2(target, norm="ortho")

        if self.high_freq_boost:
            weight   = self._freq_weight(pred.shape[-2], pred.shape[-1], pred.device)
            pred_f   = pred_f   * weight
            target_f = target_f * weight

        loss = F.mse_loss(pred_f.real, target_f.real) \
             + F.mse_loss(pred_f.imag, target_f.imag)
        return loss
