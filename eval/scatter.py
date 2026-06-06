"""
Scatter probe : z → (θ, ω) — valeurs réelles vs prédites.

Mode simple (un modèle) :
  Grille 2×2 : (θ, ω) × (Linéaire, MLP)

Mode comparaison (--compare) :
  Grille 2×4 : (θ, ω) lignes × (JEPA Lin, JEPA MLP, AE Lin, AE MLP) colonnes

Usage :
  python3 eval/scatter.py --model jepa --save visuals/scatter_jepa.png
  python3 eval/scatter.py --model rec  --save visuals/scatter_ae.png

  python3 eval/scatter.py --compare --save visuals/separability.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader, random_split

from data.dataset import PendulumSeqDataset


DARK  = "#0e0e0e"
DARK2 = "#161616"
GRID  = "#282828"
WHITE = "#e4e4e4"
DIM   = "#555555"
C_JEPA = "#4fc3f7"
C_REC  = "#ff8a65"
C_DIAG = "#888888"


# ── Chargement ────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_jepa(path, device):
    from models.jepa.model import LeWorldModel
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
    print(f"JEPA  epoch={ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss', float('nan')):.5f}")
    return model


def load_rec(path, device):
    from models.rec.model import LeWorldModelRec
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    a     = ckpt.get("args", {})
    model = LeWorldModelRec(
        embed_dim       = a.get("embed_dim",    128),
        hidden_dim      = a.get("hidden_dim",   512),
        lam             = a.get("lam",          0.5),
        n_proj          = a.get("n_proj",       512),
        rollout_k       = a.get("rollout_k",    10),
        perceptual_coef = 0.0,
        freq_coef       = 0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"AE    epoch={ckpt.get('epoch','?')}  val_loss={ckpt.get('val_loss', float('nan')):.5f}")
    return model


# ── Extraction ────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect(model, loader, device):
    Zs, Ss = [], []
    for frames, states in loader:
        z = model.encode(frames.to(device))
        Zs.append(z.cpu().numpy().reshape(-1, z.shape[-1]))
        Ss.append(states.numpy().reshape(-1, states.shape[-1]))
    return np.concatenate(Zs), np.concatenate(Ss)


# ── Probes ────────────────────────────────────────────────────────────────────

def lstsq_probe(Z_tr, S_tr, Z_va):
    mu    = Z_tr.mean(0)
    sigma = Z_tr.std(0) + 1e-8
    Zn_tr = (Z_tr - mu) / sigma
    Zn_va = (Z_va - mu) / sigma
    W, _, _, _ = np.linalg.lstsq(
        np.c_[Zn_tr, np.ones(len(Zn_tr))], S_tr, rcond=None
    )
    return np.c_[Zn_va, np.ones(len(Zn_va))] @ W


def train_mlp(Z_tr, S_tr, Z_va, device, n_epochs=200, lr=1e-3, batch_size=1024):
    from torch.utils.data import TensorDataset, DataLoader as DL
    D = Z_tr.shape[1]
    mlp = nn.Sequential(
        nn.Linear(D, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 2),
    ).to(device)
    ds     = TensorDataset(torch.from_numpy(Z_tr).float(), torch.from_numpy(S_tr).float())
    loader = DL(ds, batch_size=batch_size, shuffle=True)
    opt    = optim.Adam(mlp.parameters(), lr=lr)
    for _ in range(n_epochs):
        mlp.train()
        for z_b, s_b in loader:
            z_b, s_b = z_b.to(device), s_b.to(device)
            opt.zero_grad()
            F.mse_loss(mlp(z_b), s_b).backward()
            opt.step()
    mlp.eval()
    with torch.no_grad():
        return mlp(torch.from_numpy(Z_va).float().to(device)).cpu().numpy()


def r2(pred, true):
    ss_res = ((true - pred) ** 2).sum()
    ss_tot = ((true - true.mean()) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-8))


# ── Panneaux ──────────────────────────────────────────────────────────────────

def draw_panel(ax, true, pred, title, color):
    ax.set_facecolor(DARK2)
    for sp in ax.spines.values():
        sp.set_edgecolor(GRID)
    ax.tick_params(colors=DIM, labelsize=7)

    lo = min(true.min(), pred.min())
    hi = max(true.max(), pred.max())
    mg = (hi - lo) * 0.06
    ax.plot([lo - mg, hi + mg], [lo - mg, hi + mg],
            color=C_DIAG, lw=1.0, ls="--", zorder=1)
    ax.scatter(true, pred, s=2.5, alpha=0.20, color=color,
               rasterized=True, linewidths=0, zorder=2)

    score = r2(pred, true)
    ax.set_title(f"{title}   R²={score:.3f}", color=WHITE, fontsize=8.5, pad=4)
    ax.set_xlabel("réel", color=DIM, fontsize=7, labelpad=2)
    ax.set_ylabel("prédit", color=DIM, fontsize=7, labelpad=2)
    return score


def run_probes(model, loader, device, epochs_mlp=400):
    Z_tr, S_tr = collect(model, loader["train"], device)
    Z_va, S_va = collect(model, loader["val"],   device)
    print(f"  {Z_tr.shape[0]:,} pts train / {Z_va.shape[0]:,} pts val", flush=True)
    print("  probe linéaire…", end=" ", flush=True)
    lin_pred = lstsq_probe(Z_tr, S_tr, Z_va)
    print("probe MLP…", end=" ", flush=True)
    mlp_pred = train_mlp(Z_tr, S_tr, Z_va, device, n_epochs=epochs_mlp)
    print("done")
    return S_va, lin_pred, mlp_pred


# ── Figure mode simple ────────────────────────────────────────────────────────

def figure_single(S_va, lin_pred, mlp_pred, model_name, color, save):
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), facecolor=DARK)
    fig.patch.set_facecolor(DARK)

    labels = ["θ (rad)", "ω (rad/s)"]
    r2s    = {}
    for row, (lbl, i) in enumerate(zip(labels, [0, 1])):
        r2s[f"lin_{lbl}"] = draw_panel(axes[row, 0], S_va[:, i], lin_pred[:, i],
                                        f"{lbl}  —  Linéaire", color)
        r2s[f"mlp_{lbl}"] = draw_panel(axes[row, 1], S_va[:, i], mlp_pred[:, i],
                                        f"{lbl}  —  MLP", color)

    fig.suptitle(f"Séparabilité de l'espace latent — {model_name}",
                 color=WHITE, fontsize=11, y=1.01)
    plt.tight_layout()
    _save_or_show(fig, save)


# ── Figure mode comparaison ───────────────────────────────────────────────────

def figure_compare(S_j, lin_j, mlp_j, S_r, lin_r, mlp_r, save):
    fig = plt.figure(figsize=(15, 7.5), facecolor=DARK)
    fig.patch.set_facecolor(DARK)

    # 2 groupes séparés par un espace
    outer = gridspec.GridSpec(
        1, 2, figure=fig,
        wspace=0.10,
        left=0.05, right=0.97, top=0.88, bottom=0.10,
    )
    inner_j = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[0], hspace=0.42, wspace=0.35)
    inner_r = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[1], hspace=0.42, wspace=0.35)

    labels = ["θ (rad)", "ω (rad/s)"]
    scores = {}

    for row, (lbl, i) in enumerate(zip(labels, [0, 1])):
        ax = fig.add_subplot(inner_j[row, 0])
        scores[f"jepa_lin_{lbl}"] = draw_panel(ax, S_j[:, i], lin_j[:, i],
                                                f"{lbl}  Lin", C_JEPA)
        ax = fig.add_subplot(inner_j[row, 1])
        scores[f"jepa_mlp_{lbl}"] = draw_panel(ax, S_j[:, i], mlp_j[:, i],
                                                f"{lbl}  MLP", C_JEPA)
        ax = fig.add_subplot(inner_r[row, 0])
        scores[f"ae_lin_{lbl}"]   = draw_panel(ax, S_r[:, i], lin_r[:, i],
                                                f"{lbl}  Lin", C_REC)
        ax = fig.add_subplot(inner_r[row, 1])
        scores[f"ae_mlp_{lbl}"]   = draw_panel(ax, S_r[:, i], mlp_r[:, i],
                                                f"{lbl}  MLP", C_REC)

    # Titres de groupe
    for x, label, color in [(0.26, "JEPA", C_JEPA), (0.74, "AE", C_REC)]:
        fig.text(x, 0.935, label, ha="center", color=color,
                 fontsize=13, fontweight="bold")

    fig.suptitle("Séparabilité de l'espace latent  ·  probe linéaire & MLP  ·  z → (θ, ω)",
                 color=WHITE, fontsize=11, y=0.98)

    _save_or_show(fig, save)

    # Résumé console
    print(f"\n{'':20s}  {'JEPA Lin':>10s}  {'JEPA MLP':>10s}  {'AE Lin':>10s}  {'AE MLP':>10s}")
    print("─" * 66)
    for lbl in labels:
        print(f"  {lbl:18s}  "
              f"{scores[f'jepa_lin_{lbl}']:10.3f}  "
              f"{scores[f'jepa_mlp_{lbl}']:10.3f}  "
              f"{scores[f'ae_lin_{lbl}']:10.3f}  "
              f"{scores[f'ae_mlp_{lbl}']:10.3f}")


def _save_or_show(fig, save):
    plt.tight_layout()
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save, dpi=150, bbox_inches="tight", facecolor=DARK)
        print(f"\nSauvegardé : {save}")
    else:
        plt.show()
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def make_loaders(dataset_dir, batch_size=16):
    ds    = PendulumSeqDataset(dataset_dir)
    n_val = max(1, int(len(ds) * 0.15))
    tr_ds, va_ds = random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    return {
        "train": DataLoader(tr_ds, batch_size=batch_size, shuffle=True,  num_workers=0),
        "val":   DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=0),
    }


def main(args):
    if args.save:
        matplotlib.use("Agg")

    device  = get_device()
    loaders = make_loaders(args.dataset_dir)
    print(f"Device : {device}")

    if args.compare:
        print("\n── JEPA ─────────────────────────────────────────────")
        jepa = load_jepa(args.jepa_ckpt, device)
        S_j, lin_j, mlp_j = run_probes(jepa, loaders, device, args.epochs_mlp)
        del jepa

        print("\n── AE ───────────────────────────────────────────────")
        rec = load_rec(args.rec_ckpt, device)
        S_r, lin_r, mlp_r = run_probes(rec, loaders, device, args.epochs_mlp)
        del rec

        figure_compare(S_j, lin_j, mlp_j, S_r, lin_r, mlp_r, args.save)

    else:
        if args.model == "jepa":
            model = load_jepa(args.jepa_ckpt, device)
            color = C_JEPA
            name  = "JEPA"
        else:
            model = load_rec(args.rec_ckpt, device)
            color = C_REC
            name  = "AE"

        S_va, lin_pred, mlp_pred = run_probes(model, loaders, device, args.epochs_mlp)
        del model
        figure_single(S_va, lin_pred, mlp_pred, name, color, args.save)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",       choices=["jepa", "rec"], default="jepa",
                   help="Modèle (ignoré si --compare)")
    p.add_argument("--compare",     action="store_true",
                   help="Compare JEPA vs AE en 2×4")
    p.add_argument("--jepa-ckpt",   default="checkpoints/jepa/lewm_best.pt")
    p.add_argument("--rec-ckpt",    default="checkpoints/rec/lewm_rec_best.pt")
    p.add_argument("--dataset-dir", default="dataset/pendulum")
    p.add_argument("--epochs-mlp",  type=int, default=400)
    p.add_argument("--save",        default=None)
    main(p.parse_args())
