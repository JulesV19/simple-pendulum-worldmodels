"""
Probe linéaire z → (θ, ω) — comparaison LeWorldModel (JEPA) vs LeWorldModelRec (AE).

Principe : on gèle l'encodeur, on entraîne une régression linéaire stricte
z → (θ, ω) sur les labels du dataset. Le R² mesure combien d'information
sur l'état physique est linéairement accessible dans z.

  R²(ω) élevé → le modèle a encodé la vitesse angulaire dans z
              → le monde model a compris la dynamique, pas juste la position

Usage :
  # Probe unique
  python3 probe.py --model jepa --checkpoint checkpoints/jepa/lewm_best.pt
  python3 probe.py --model rec  --checkpoint checkpoints/rec/lewm_rec_best.pt

  # Sample efficiency : probe entraîné sur N% des trajectoires labellisées
  python3 probe.py --model jepa --checkpoint checkpoints/jepa/lewm_best.pt --label-frac 0.1

  # Sweep sur un dossier de checkpoints → courbe R² vs epoch
  python3 probe.py --model jepa --checkpoint-dir checkpoints/jepa/ --plot
  python3 probe.py --model rec  --checkpoint-dir checkpoints/rec/  --plot

  # Comparaison directe des deux meilleurs checkpoints
  python3 probe.py --compare \\
      --jepa-ckpt checkpoints/jepa/lewm_best.pt \\
      --rec-ckpt  checkpoints/rec/lewm_rec_best.pt
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset, random_split

from models.jepa.model import LeWorldModel
from models.rec.model import LeWorldModelRec
from data.dataset import PendulumSeqDataset


# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# ── Chargement des encodeurs ──────────────────────────────────────────────────

def load_jepa(path: str, device) -> LeWorldModel:
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    a     = ckpt.get("args", {})
    model = LeWorldModel(
        embed_dim    = a.get("embed_dim",    128),
        hidden_dim   = a.get("hidden_dim",   512),
        lam          = a.get("lam",          0.5),
        n_proj       = a.get("n_proj",       512),
        ema_momentum = a.get("ema_momentum", 0.996),
        rollout_k    = a.get("rollout_k",    10),
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt.get("epoch", 0)


def load_rec(path: str, device) -> LeWorldModelRec:
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    a     = ckpt.get("args", {})
    model = LeWorldModelRec(
        embed_dim       = a.get("embed_dim",       128),
        hidden_dim      = a.get("hidden_dim",      512),
        lam             = a.get("lam",             0.5),
        n_proj          = a.get("n_proj",          512),
        rollout_k       = a.get("rollout_k",       10),
        perceptual_coef = 0.0,
        freq_coef       = 0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, ckpt.get("epoch", 0)


def load_encoder(model_type: str, path: str, device):
    if model_type == "jepa":
        return load_jepa(path, device)
    elif model_type == "rec":
        return load_rec(path, device)
    else:
        raise ValueError(f"model_type doit être 'jepa' ou 'rec', reçu : {model_type}")


# ── Extraction des embeddings ─────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(model, dataset_indices, dataset_dir: str,
                       device, chunk: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Encode toutes les frames de dataset_indices (indices dans PendulumSeqDataset).
    Retourne :
      Z      : (N_frames, D)  float32
      labels : (N_frames, 2)  float32   [θ, ω]

    On skippe t=0 pour chaque trajectoire : à t=0 le diff de frames est nul
    donc ω n'est pas encodé dans z[0].
    """
    full_ds = PendulumSeqDataset(dataset_dir, seq_len=None)  # trajectoires complètes
    Z_list, L_list = [], []

    for idx in dataset_indices:
        frames, states = full_ds[idx]          # (T, 3, H, W), (T, 2)
        T = frames.shape[0]

        # Encoder par chunks pour éviter l'OOM
        z_chunks = []
        for start in range(0, T, chunk):
            f_chunk = frames[start:start + chunk].unsqueeze(0).to(device)  # (1, chunk, 3, H, W)
            z_chunk = model.encode(f_chunk)                                # (1, chunk, D)
            z_chunks.append(z_chunk[0].cpu())
        z_all = torch.cat(z_chunks, dim=0)     # (T, D)

        # Skip t=0 (diff=0 → ω absent de z)
        Z_list.append(z_all[1:])
        L_list.append(states[1:])

    Z      = torch.cat(Z_list, dim=0).float()   # (N, D)
    labels = torch.cat(L_list, dim=0).float()   # (N, 2)
    return Z, labels


# ── Probe linéaire & MLP ──────────────────────────────────────────────────────

def r2_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """R² par dimension — shape (D,)"""
    ss_res = ((y_true - y_pred) ** 2).sum(0)
    ss_tot = ((y_true - y_true.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / ss_tot.clamp(min=1e-8)


class _MLPProbe(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return self.net(x)


def _train_one_probe(probe: nn.Module,
                     Z_train_n: torch.Tensor, L_train: torch.Tensor,
                     Z_val_n:   torch.Tensor, L_val:   torch.Tensor,
                     epochs: int, lr: float, batch_size: int,
                     device) -> dict:
    opt    = optim.Adam(probe.parameters(), lr=lr)
    ds     = TensorDataset(Z_train_n.to(device), L_train.to(device))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    for _ in range(epochs):
        probe.train()
        for z_b, l_b in loader:
            loss = nn.functional.mse_loss(probe(z_b), l_b)
            opt.zero_grad()
            loss.backward()
            opt.step()

    probe.eval()
    with torch.no_grad():
        pred = probe(Z_val_n.to(device)).cpu()

    r2 = r2_score(pred, L_val)
    return {"r2_theta": r2[0].item(), "r2_omega": r2[1].item(), "r2_mean": r2.mean().item()}


def train_probe(Z_train: torch.Tensor, L_train: torch.Tensor,
                Z_val:   torch.Tensor, L_val:   torch.Tensor,
                epochs: int = 200, lr: float = 1e-2,
                batch_size: int = 1024, device=None) -> dict:
    """
    Entraîne un probe linéaire ET un probe MLP z → (θ, ω).
    Retourne un dict avec les R² des deux probes.
    L'écart linear→MLP mesure si l'information est accessible linéairement
    ou seulement de façon non-linéaire (important pour valider la comparaison).
    """
    if device is None:
        device = torch.device("cpu")

    z_mean = Z_train.mean(0)
    z_std  = Z_train.std(0).clamp(min=1e-6)
    Z_train_n = (Z_train - z_mean) / z_std
    Z_val_n   = (Z_val   - z_mean) / z_std

    D = Z_train_n.shape[1]

    lin_probe = nn.Linear(D, 2).to(device)
    lin_res = _train_one_probe(lin_probe, Z_train_n, L_train, Z_val_n, L_val,
                               epochs, lr, batch_size, device)

    mlp_probe = _MLPProbe(D).to(device)
    mlp_res = _train_one_probe(mlp_probe, Z_train_n, L_train, Z_val_n, L_val,
                               epochs, lr * 0.1, batch_size, device)

    return {
        "r2_theta":     lin_res["r2_theta"],
        "r2_omega":     lin_res["r2_omega"],
        "r2_mean":      lin_res["r2_mean"],
        "mlp_r2_theta": mlp_res["r2_theta"],
        "mlp_r2_omega": mlp_res["r2_omega"],
        "mlp_r2_mean":  mlp_res["r2_mean"],
    }


# ── Probe complète sur un checkpoint ─────────────────────────────────────────

def run_probe(model_type: str, ckpt_path: str,
              dataset_dir: str, label_frac: float,
              probe_epochs: int, probe_lr: float, probe_bs: int,
              device, seed: int = 42, val_split: float = 0.1,
              verbose: bool = True) -> dict:
    """
    Pipeline complet pour un checkpoint :
      1. Charge l'encodeur
      2. Sépare train/val (même split que l'entraînement du modèle)
      3. Encode les frames (encodeur gelé)
      4. Entraîne le probe sur label_frac des trajectoires train
      5. Évalue sur val
    """
    model, epoch = load_encoder(model_type, ckpt_path, device)

    # Même split train/val que make_seq_dataloaders
    full_ds = PendulumSeqDataset(dataset_dir, seq_len=None)
    n       = len(full_ds)
    n_val   = int(n * val_split)
    n_train = n - n_val
    gen     = torch.Generator().manual_seed(seed)
    train_idx, val_idx = random_split(range(n), [n_train, n_val], generator=gen)
    train_idx = list(train_idx)
    val_idx   = list(val_idx)

    # Sous-échantillonnage pour la sample efficiency
    if label_frac < 1.0:
        rng     = np.random.default_rng(seed)
        n_probe = max(1, int(len(train_idx) * label_frac))
        train_idx = rng.choice(train_idx, n_probe, replace=False).tolist()

    if verbose:
        print(f"  Encodage {len(train_idx)} traj train + {len(val_idx)} traj val…", end=" ", flush=True)

    Z_train, L_train = extract_embeddings(model, train_idx, dataset_dir, device)
    Z_val,   L_val   = extract_embeddings(model, val_idx,   dataset_dir, device)

    if verbose:
        print(f"Z_train={Z_train.shape}  Z_val={Z_val.shape}")
        print(f"  Entraînement du probe ({probe_epochs} epochs)…", end=" ", flush=True)

    results = train_probe(Z_train, L_train, Z_val, L_val,
                          epochs=probe_epochs, lr=probe_lr,
                          batch_size=probe_bs, device=torch.device("cpu"))
    results["epoch"] = epoch
    results["ckpt"]  = str(ckpt_path)

    if verbose:
        print(f"  Linear  R²(θ)={results['r2_theta']:.4f}  R²(ω)={results['r2_omega']:.4f}  R²(mean)={results['r2_mean']:.4f}")
        print(f"  MLP     R²(θ)={results['mlp_r2_theta']:.4f}  R²(ω)={results['mlp_r2_omega']:.4f}  R²(mean)={results['mlp_r2_mean']:.4f}")
        gap = results['mlp_r2_mean'] - results['r2_mean']
        print(f"  Gap lin→MLP : {gap:+.4f}  {'(info non-linéaire présente)' if gap > 0.01 else '(embeddings linéairement structurés)'}")

    return results


# ── Mode sweep (checkpoint_dir) ───────────────────────────────────────────────

def run_sweep(model_type: str, ckpt_dir: str, dataset_dir: str,
              label_frac: float, probe_epochs: int, probe_lr: float,
              probe_bs: int, device, plot: bool, seed: int = 42) -> list[dict]:
    ckpts = sorted(Path(ckpt_dir).glob("*.pt"),
                   key=lambda p: torch.load(p, map_location="cpu",
                                            weights_only=False).get("epoch", 0))
    if not ckpts:
        raise FileNotFoundError(f"Aucun .pt trouvé dans {ckpt_dir}")

    print(f"Sweep sur {len(ckpts)} checkpoints dans {ckpt_dir}")
    all_results = []
    for ckpt in ckpts:
        ep = torch.load(ckpt, map_location="cpu", weights_only=False).get("epoch", "?")
        print(f"\n[epoch {ep}] {ckpt.name}")
        r = run_probe(model_type, str(ckpt), dataset_dir, label_frac,
                      probe_epochs, probe_lr, probe_bs, device, seed=seed)
        all_results.append(r)

    if plot:
        _plot_sweep(all_results, model_type, ckpt_dir)

    return all_results


def _plot_sweep(results: list[dict], model_type: str, ckpt_dir: str):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib non disponible — skip plot")
        return

    epochs  = [r["epoch"]     for r in results]
    r2_th   = [r["r2_theta"]  for r in results]
    r2_om   = [r["r2_omega"]  for r in results]
    r2_mean = [r["r2_mean"]   for r in results]

    DARK = "#111"
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(DARK)
    ax.set_facecolor(DARK)

    ax.plot(epochs, r2_th,   color="#4fc3f7", lw=2, marker="o", ms=4, label="R²(θ)")
    ax.plot(epochs, r2_om,   color="#ff8a65", lw=2, marker="o", ms=4, label="R²(ω)")
    ax.plot(epochs, r2_mean, color="#a5d6a7", lw=2, marker="o", ms=4, label="R²(mean)", ls="--")

    ax.set_xlabel("epoch", color="white")
    ax.set_ylabel("R²", color="white")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(f"Probe linéaire — {model_type.upper()} — {Path(ckpt_dir).name}",
                 color="white")
    ax.legend(facecolor="#222", labelcolor="white", edgecolor="#444")
    ax.tick_params(colors="white")
    ax.axhline(1.0, color="#555", ls=":", lw=1)
    for sp in ax.spines.values():
        sp.set_edgecolor("#444")

    out = Path(ckpt_dir) / f"probe_sweep_{model_type}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120, bbox_inches="tight", facecolor=DARK)
    plt.show()
    print(f"Plot sauvegardé : {out}")


# ── Mode comparaison directe ──────────────────────────────────────────────────

def run_compare(jepa_ckpt: str, rec_ckpt: str, dataset_dir: str,
                label_frac: float, probe_epochs: int, probe_lr: float,
                probe_bs: int, device, plot: bool, seed: int = 42):

    print("─" * 55)
    print("JEPA")
    print("─" * 55)
    r_jepa = run_probe("jepa", jepa_ckpt, dataset_dir, label_frac,
                       probe_epochs, probe_lr, probe_bs, device, seed=seed)

    print("\n" + "─" * 55)
    print("AE (LeWorldModelRec)")
    print("─" * 55)
    r_rec = run_probe("rec", rec_ckpt, dataset_dir, label_frac,
                      probe_epochs, probe_lr, probe_bs, device, seed=seed)

    print("\n" + "═" * 65)
    print(f"{'':25s}  {'JEPA':>10s}  {'AE (Rec)':>10s}  {'Δ(JEPA-AE)':>10s}")
    print("─" * 65)
    print("  — Probe LINÉAIRE —")
    for key, label in [("r2_theta","R²(θ)"), ("r2_omega","R²(ω)"), ("r2_mean","R²(mean)")]:
        delta = r_jepa[key] - r_rec[key]
        print(f"  {label:22s}  {r_jepa[key]:10.4f}  {r_rec[key]:10.4f}  {delta:+10.4f}")
    print("─" * 65)
    print("  — Probe MLP (2×256) —")
    for key, label in [("mlp_r2_theta","R²(θ)"), ("mlp_r2_omega","R²(ω)"), ("mlp_r2_mean","R²(mean)")]:
        delta = r_jepa[key] - r_rec[key]
        print(f"  {label:22s}  {r_jepa[key]:10.4f}  {r_rec[key]:10.4f}  {delta:+10.4f}")
    print("─" * 65)
    print("  — Gap linéaire → MLP (info non-linéaire) —")
    gap_jepa = r_jepa["mlp_r2_mean"] - r_jepa["r2_mean"]
    gap_rec  = r_rec["mlp_r2_mean"]  - r_rec["r2_mean"]
    print(f"  {'Gap R²(mean)':22s}  {gap_jepa:+10.4f}  {gap_rec:+10.4f}")
    print("═" * 65)
    winner_lin = "JEPA" if r_jepa["r2_mean"] > r_rec["r2_mean"] else "AE"
    winner_mlp = "JEPA" if r_jepa["mlp_r2_mean"] > r_rec["mlp_r2_mean"] else "AE"
    print(f"  Linéaire : {winner_lin} gagne  |  MLP : {winner_mlp} gagne")
    print("═" * 65)

    if plot:
        _plot_compare(r_jepa, r_rec)


def _plot_compare(r_jepa: dict, r_rec: dict):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    DARK = "#111"
    metrics = ["R²(θ)", "R²(ω)", "R²(mean)"]
    keys_lin = ["r2_theta",     "r2_omega",     "r2_mean"]
    keys_mlp = ["mlp_r2_theta", "mlp_r2_omega", "mlp_r2_mean"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    fig.patch.set_facecolor(DARK)
    titles = ["Probe LINÉAIRE", "Probe MLP (2×256)"]

    for ax, title, k_lin, k_mlp in [
        (axes[0], titles[0], keys_lin, keys_lin),
        (axes[1], titles[1], keys_mlp, keys_mlp),
    ]:
        ax.set_facecolor(DARK)
        x = np.arange(len(metrics))
        width = 0.35
        vals_j = [r_jepa[k] for k in (k_lin if title == titles[0] else k_mlp)]
        vals_r = [r_rec[k]  for k in (k_lin if title == titles[0] else k_mlp)]

        ax.bar(x - width/2, vals_j, width, label="JEPA",     color="#4fc3f7", alpha=0.85)
        ax.bar(x + width/2, vals_r, width, label="AE (Rec)", color="#ff8a65", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, color="white")
        ax.set_ylabel("R²", color="white")
        ax.set_ylim(0, 1.08)
        ax.set_title(title, color="white")
        ax.legend(facecolor="#222", labelcolor="white", edgecolor="#444")
        ax.tick_params(colors="white")
        ax.axhline(1.0, color="#555", ls=":", lw=1)
        for sp in ax.spines.values():
            sp.set_edgecolor("#444")

    fig.suptitle("Probe z → (θ, ω) — JEPA vs AE", color="white", fontsize=13)
    plt.tight_layout()
    plt.savefig("visuals/probe_compare.png", dpi=120, bbox_inches="tight", facecolor=DARK)
    plt.show()
    print("Plot sauvegardé : visuals/probe_compare.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Probe linéaire z → (θ, ω) pour JEPA vs AE"
    )

    # Mode
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--model",           choices=["jepa", "rec"],
                      help="Type de modèle (mode probe unique ou sweep)")
    mode.add_argument("--compare",         action="store_true",
                      help="Mode comparaison directe JEPA vs AE")

    # Checkpoints
    parser.add_argument("--checkpoint",     default=None,
                        help="Chemin vers un checkpoint unique")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Dossier de checkpoints pour le sweep")
    parser.add_argument("--jepa-ckpt",      default="checkpoints/jepa/lewm_best.pt")
    parser.add_argument("--rec-ckpt",       default="checkpoints/rec/lewm_rec_best.pt")

    # Dataset
    parser.add_argument("--dataset-dir",    default="dataset/pendulum")

    # Probe
    parser.add_argument("--label-frac",     type=float, default=1.0,
                        help="Fraction des trajectoires train utilisées pour le probe (0-1)")
    parser.add_argument("--probe-epochs",   type=int,   default=200)
    parser.add_argument("--probe-lr",       type=float, default=1e-2)
    parser.add_argument("--probe-bs",       type=int,   default=1024)

    # Misc
    parser.add_argument("--plot",           action="store_true")
    parser.add_argument("--seed",           type=int,   default=42)

    args = parser.parse_args()
    device = get_device()
    print(f"Device : {device}")
    Path("visuals").mkdir(parents=True, exist_ok=True)

    if args.compare:
        run_compare(
            args.jepa_ckpt, args.rec_ckpt,
            args.dataset_dir, args.label_frac,
            args.probe_epochs, args.probe_lr, args.probe_bs,
            device, args.plot, args.seed,
        )

    elif args.checkpoint_dir:
        run_sweep(
            args.model, args.checkpoint_dir,
            args.dataset_dir, args.label_frac,
            args.probe_epochs, args.probe_lr, args.probe_bs,
            device, args.plot, args.seed,
        )

    elif args.checkpoint:
        print(f"Probe [{args.model.upper()}] sur {args.checkpoint}")
        if args.label_frac < 1.0:
            print(f"  label_frac={args.label_frac:.0%} des trajectoires train")
        run_probe(
            args.model, args.checkpoint,
            args.dataset_dir, args.label_frac,
            args.probe_epochs, args.probe_lr, args.probe_bs,
            device, args.seed,
        )

    else:
        parser.error("--checkpoint ou --checkpoint-dir requis avec --model")


if __name__ == "__main__":
    main()
