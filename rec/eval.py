"""
Évaluation LeWorldModelRec.

  1. Linear probe      — R² lstsq exact  z → (θ, ω)
  2. MLP probe         — R² non-linéaire z → (θ, ω)
  3. Qualité pixel     — MSE & PSNR reconstruction + prédiction horizon
  4. Uniformité &      — Détecte le collapse, cohérence temporelle
     Alignement
  5. Horizon latent    — Cosine similarity z_pred / z_réel à t+1..t+10
  6. Horizon pixel     — MSE decode(z_pred) / frame_réelle à t+1..t+10

Usage:
  python3 rec/eval.py --checkpoint checkpoints/rec/lewm_rec_best.pt
  python3 rec/eval.py --checkpoint checkpoints/rec/lewm_rec_best.pt --save visuals/eval_rec.png
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader, TensorDataset, random_split

from models.rec.model import LeWorldModelRec
from data.dataset import PendulumSeqDataset


DARK        = "#111"
STATE_NAMES = ["θ", "ω"]


# ── Setup ──────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_model(path: str, device) -> LeWorldModelRec:
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    a     = ckpt.get("args", {})
    model = LeWorldModelRec(
        embed_dim       = a.get("embed_dim",       128),
        hidden_dim      = a.get("hidden_dim",      512),
        lam             = a.get("lam",             0.5),
        n_proj          = a.get("n_proj",          512),
        rollout_k       = a.get("rollout_k",       10),
        perceptual_coef = 0.0,   # désactivé à l'inférence
        freq_coef       = 0.0,
    ).to(device)
    missing, _ = model.load_state_dict(ckpt["model"], strict=False)
    if missing:
        print(f"  [info] clés absentes : {missing[:3]}…")
    model.eval()
    print(f"LeWorldModelRec : epoch={ckpt.get('epoch','?')}  "
          f"val_loss={ckpt.get('val_loss', float('nan')):.5f}")
    return model


def make_loaders(dataset_dir: str, val_split=0.1, seed=42):
    ds    = PendulumSeqDataset(dataset_dir)
    n_val = int(len(ds) * val_split)
    train_ds, val_ds = random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(seed),
    )
    return (DataLoader(train_ds, batch_size=8, shuffle=True,  num_workers=0),
            DataLoader(val_ds,   batch_size=8, shuffle=False, num_workers=0))


# ── Collecte des embeddings ────────────────────────────────────────────────────

@torch.no_grad()
def collect_embeddings(model, loader, device, normalize=False):
    all_z, all_s, seqs = [], [], []
    for frames, states in loader:
        z = model.encode(frames.to(device))     # (B, T, D)
        if normalize:
            z = F.normalize(z, dim=-1)
        z_np = z.cpu().numpy()
        s_np = states.numpy()
        for b in range(len(z_np)):
            all_z.append(z_np[b])
            all_s.append(s_np[b])
            seqs.append(z_np[b])
    return np.concatenate(all_z), np.concatenate(all_s), seqs


# ── Utilitaire R² ──────────────────────────────────────────────────────────────

def compute_r2s(preds: np.ndarray, targets: np.ndarray):
    r2s = []
    for i in range(targets.shape[1]):
        ss_res = ((targets[:, i] - preds[:, i]) ** 2).sum()
        ss_tot = ((targets[:, i] - targets[:, i].mean()) ** 2).sum()
        r2s.append(float(1 - ss_res / (ss_tot + 1e-8)))
    return r2s


# ── 1. Linear probe (lstsq) + MLP probe ───────────────────────────────────────

def _lstsq_probe(Z_tr, S_tr, Z_vl):
    """Régression linéaire exacte (moindres carrés). Solution optimale garantie."""
    mu    = Z_tr.mean(0)
    sigma = Z_tr.std(0) + 1e-8
    Zn_tr = (Z_tr - mu) / sigma
    Zn_vl = (Z_vl - mu) / sigma
    W, _, _, _ = np.linalg.lstsq(
        np.c_[Zn_tr, np.ones(len(Zn_tr))], S_tr, rcond=None
    )
    return np.c_[Zn_vl, np.ones(len(Zn_vl))] @ W


def _run_mlp_probe(head, Zt, St, Zv, n_epochs, lr=3e-4, batch_size=1024):
    opt    = optim.Adam(head.parameters(), lr=lr)
    loader = DataLoader(TensorDataset(Zt, St), batch_size=batch_size, shuffle=True)
    for _ in range(n_epochs):
        head.train()
        for zb, sb in loader:
            opt.zero_grad()
            F.mse_loss(head(zb), sb).backward()
            opt.step()
    head.eval()
    with torch.no_grad():
        return head(Zv).cpu().numpy()


def linear_probe(model, train_loader, val_loader, device, n_epochs=200):
    print("\n── Linear probe  vs  MLP probe ──────────────────────────")
    Z_tr, S_tr, _ = collect_embeddings(model, train_loader, device)
    Z_va, S_va, _ = collect_embeddings(model, val_loader,   device)

    D        = Z_tr.shape[1]
    n_states = S_tr.shape[1]

    lin_preds = _lstsq_probe(Z_tr, S_tr, Z_va)
    r2s_lin   = compute_r2s(lin_preds, S_va)

    Zt = torch.from_numpy(Z_tr).float().to(device)
    St = torch.from_numpy(S_tr).float().to(device)
    Zv = torch.from_numpy(Z_va).float().to(device)
    mlp = nn.Sequential(
        nn.Linear(D, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, n_states),
    ).to(device)
    mlp_preds = _run_mlp_probe(mlp, Zt, St, Zv, n_epochs)
    r2s_mlp   = compute_r2s(mlp_preds, S_va)

    r2_lin = float(np.mean(r2s_lin))
    r2_mlp = float(np.mean(r2s_mlp))

    print(f"  {'':6}  {'Linéaire':>10}  {'MLP':>10}")
    for name, rl, rm in zip(STATE_NAMES, r2s_lin, r2s_mlp):
        print(f"  R²({name})  {rl:>10.4f}  {rm:>10.4f}")
    print(f"  {'global':6}  {r2_lin:>10.4f}  {r2_mlp:>10.4f}")

    gap = r2_mlp - r2_lin
    if gap > 0.05:
        print(f"  → Info présente mais non-linéaire (gap = +{gap:.3f})")
    else:
        print(f"  → Peu de gain non-linéaire (gap = {gap:+.3f})")

    return r2s_lin, r2_lin, lin_preds, S_va, r2s_mlp, r2_mlp


# ── 2. Qualité pixel (reconstruction + prédiction) ────────────────────────────

@torch.no_grad()
def pixel_quality(model: LeWorldModelRec, val_loader, device):
    """
    MSE et PSNR en pixel space pour :
      - Reconstruction : decode(encode(frame_t)) vs frame_t
      - Prédiction t+1 : decode(predictor(encode(frame_t))) vs frame_{t+1}
    """
    print("\n── Qualité pixel ─────────────────────────────────────────")
    rec_mses, pred_mses = [], []

    for frames, _ in val_loader:
        frames = frames.to(device)              # (B, T, 3, H, W)
        B, T   = frames.shape[:2]

        z        = model.encode(frames)         # (B, T, D)
        recon    = model.decode(z)              # (B, T, 3, H, W)
        rec_mse  = F.mse_loss(recon, frames).item()
        rec_mses.append(rec_mse)

        if T > 1:
            z_pred    = model.predictor(z[:, :-1])          # (B, T-1, D)
            pred_dec  = model.decode(z_pred)                # (B, T-1, 3, H, W)
            pred_mse  = F.mse_loss(pred_dec, frames[:, 1:]).item()
            pred_mses.append(pred_mse)

    rec_mse  = float(np.mean(rec_mses))
    pred_mse = float(np.mean(pred_mses)) if pred_mses else float("nan")

    def psnr(mse): return 10 * np.log10(1.0 / (mse + 1e-10))

    print(f"  Reconstruction  MSE={rec_mse:.5f}   PSNR={psnr(rec_mse):.1f} dB")
    print(f"  Prédiction t+1  MSE={pred_mse:.5f}  PSNR={psnr(pred_mse):.1f} dB")
    return rec_mse, pred_mse


# ── 3. Uniformité & Alignement ─────────────────────────────────────────────────

def uniformity_alignment(seqs_train, seqs_val):
    print("\n── Uniformité & Alignement ───────────────────────────────")
    Z    = np.concatenate(seqs_val)
    idx  = np.random.choice(len(Z), size=min(2000, len(Z)), replace=False)
    Zs   = Z[idx]
    d2   = np.sum((Zs[:, None] - Zs[None, :]) ** 2, axis=-1)
    mask = ~np.eye(len(Zs), dtype=bool)
    uniformity = float(np.log(np.exp(-2 * d2[mask]).mean() + 1e-8))

    align_vals = [((z[1:] - z[:-1]) ** 2).sum(axis=-1).mean() for z in seqs_train]
    alignment  = float(np.mean(align_vals))

    print(f"  Uniformité = {uniformity:.4f}  (cible : -2 à -4,  0 = collapse)")
    print(f"  Alignement = {alignment:.4f}  (cible : < 0.5)")
    return uniformity, alignment


# ── 4. Horizon latent (cos-sim) + pixel (MSE) ─────────────────────────────────

@torch.no_grad()
def prediction_horizon(model: LeWorldModelRec, val_loader, device,
                       horizons=(1, 2, 5, 10)):
    print("\n── Horizon de prédiction ─────────────────────────────────")
    lat_sims = {h: [] for h in horizons}
    pix_mses = {h: [] for h in horizons}

    for frames, _ in val_loader:
        frames = frames.to(device)
        B, T   = frames.shape[:2]
        z_all  = model.encode(frames)                            # (B, T, D)
        z_norm = F.normalize(z_all, dim=-1)

        for h in horizons:
            if T <= h:
                continue
            z_roll = z_all[:, :T - h]
            for _ in range(h):
                z_roll = model.predictor(z_roll)                 # (B, T-h, D)

            # Cosine similarity latente
            z_pred_n = F.normalize(z_roll, dim=-1)
            z_tgt_n  = z_norm[:, h:]
            lat_sims[h].append((z_pred_n * z_tgt_n).sum(-1).mean().item())

            # MSE pixel
            frames_pred = model.decode(z_roll)                   # (B, T-h, 3, H, W)
            frames_tgt  = frames[:, h:]
            pix_mses[h].append(F.mse_loss(frames_pred, frames_tgt).item())

    print(f"  {'Horizon':>8}  {'Cos-sim (z)':>13}  {'MSE pixel':>12}  {'PSNR':>8}")
    lat_out, pix_out = {}, {}
    for h in horizons:
        ls  = float(np.mean(lat_sims[h])) if lat_sims[h] else float("nan")
        pm  = float(np.mean(pix_mses[h])) if pix_mses[h] else float("nan")
        psnr = 10 * np.log10(1.0 / (pm + 1e-10))
        lat_out[h] = ls
        pix_out[h] = pm
        print(f"  t+{h:>6}  {ls:>13.4f}  {pm:>12.5f}  {psnr:>7.1f} dB")

    return lat_out, pix_out


# ── Figure de synthèse ─────────────────────────────────────────────────────────

def save_figure(r2s_lin, r2_lin, lin_preds, s_val,
                r2s_mlp, r2_mlp,
                rec_mse, pred_mse,
                uniformity, alignment,
                lat_sims, pix_mses,
                save_path):

    def psnr(mse): return 10 * np.log10(1.0 / (mse + 1e-10))

    fig = plt.figure(figsize=(16, 10), facecolor=DARK)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    def style(ax):
        ax.set_facecolor(DARK)
        ax.tick_params(colors="white")
        for sp in ax.spines.values(): sp.set_edgecolor("#444")

    C_REC = "#ff8a65"

    # ── [0,0] Probe R² ──────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0]); style(ax)
    x  = np.arange(len(STATE_NAMES))
    w  = 0.35
    ax.bar(x - w/2, r2s_lin, width=w, color=["#4fc3f7", "#a5d6a7"],
           alpha=0.85, label=f"Linéaire  R²={r2_lin:.3f}")
    ax.bar(x + w/2, r2s_mlp, width=w, color=["#4fc3f7", "#a5d6a7"],
           alpha=0.45, hatch="//", label=f"MLP       R²={r2_mlp:.3f}")
    ax.axhline(1.0, color="#555", lw=0.8, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(STATE_NAMES, color="white")
    ax.set_ylim(-0.1, 1.15)
    ax.set_title("Probe R² par état", color="white", fontsize=10)
    ax.legend(fontsize=7, labelcolor="white", facecolor="#222", edgecolor="#444")
    for i, (vl, vm) in enumerate(zip(r2s_lin, r2s_mlp)):
        ax.text(i - w/2, vl + 0.02, f"{vl:.3f}", ha="center",
                color="#4fc3f7", fontsize=7)
        ax.text(i + w/2, vm + 0.02, f"{vm:.3f}", ha="center",
                color="#888",    fontsize=7)

    # ── [0,1] Scatter θ ─────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1]); style(ax2)
    ax2.scatter(s_val[:, 0], lin_preds[:, 0], s=4, alpha=0.3, color=C_REC)
    lo = min(s_val[:, 0].min(), lin_preds[:, 0].min())
    hi = max(s_val[:, 0].max(), lin_preds[:, 0].max())
    ax2.plot([lo, hi], [lo, hi], color="#888", lw=1.2, ls="--", label="y=x")
    ax2.set_xlabel("θ réel",  color="white", fontsize=9)
    ax2.set_ylabel("θ prédit", color="white", fontsize=9)
    ax2.set_title(f"Scatter θ  (R²={r2s_lin[0]:.3f})", color="white", fontsize=10)
    ax2.legend(fontsize=8, labelcolor="white", facecolor="#222", edgecolor="#444")

    # ── [0,2] Qualité pixel ─────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2]); style(ax3)
    labels = ["Recon.", "Préd. t+1"]
    vals   = [rec_mse, pred_mse]
    colors = ["#a5d6a7", "#ffcc80"]
    bars   = ax3.bar(labels, vals, color=colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, vals):
        ax3.text(bar.get_x() + bar.get_width()/2, v + max(vals)*0.01,
                 f"MSE={v:.4f}\nPSNR={psnr(v):.1f}dB",
                 ha="center", va="bottom", color="white", fontsize=8)
    ax3.set_title("Qualité pixel (decode)", color="white", fontsize=10)
    ax3.set_ylabel("MSE", color="white", fontsize=9)
    ax3.tick_params(axis="x", colors="white")
    ax3.set_ylim(0, max(vals) * 1.35)

    # ── [1,0] Horizon latent (cos-sim) ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0]); style(ax4)
    hs   = list(lat_sims.keys())
    sims = list(lat_sims.values())
    ax4.plot(hs, sims, color=C_REC, lw=2, marker="o", markersize=7)
    ax4.fill_between(hs, sims, alpha=0.15, color=C_REC)
    ax4.axhline(1.0, color="#555", lw=0.8, ls="--")
    ax4.axhline(0.0, color="#555", lw=0.8, ls=":")
    ax4.set_xlabel("Horizon (frames)", color="white", fontsize=9)
    ax4.set_ylabel("Cosine similarity (espace latent)", color="white", fontsize=9)
    ax4.set_title("Horizon — espace latent", color="white", fontsize=10)
    ax4.set_xticks(hs); ax4.set_xticklabels([f"t+{h}" for h in hs])
    ax4.set_ylim(-0.1, 1.1)
    for h, s in zip(hs, sims):
        ax4.annotate(f"{s:.3f}", (h, s), textcoords="offset points",
                     xytext=(0, 10), ha="center", color="white", fontsize=8)

    # ── [1,1] Horizon pixel (PSNR) ──────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1]); style(ax5)
    psnrs = [10 * np.log10(1.0 / (pix_mses[h] + 1e-10)) for h in hs]
    ax5.plot(hs, psnrs, color="#ffcc80", lw=2, marker="s", markersize=7)
    ax5.fill_between(hs, psnrs, alpha=0.12, color="#ffcc80")
    # Ligne de référence : PSNR reconstruction (t=0)
    psnr_rec = psnr(rec_mse)
    ax5.axhline(psnr_rec, color="#a5d6a7", lw=1, ls="--",
                label=f"Reconstruction (t=0) {psnr_rec:.1f} dB")
    ax5.set_xlabel("Horizon (frames)", color="white", fontsize=9)
    ax5.set_ylabel("PSNR (dB)", color="white", fontsize=9)
    ax5.set_title("Horizon — espace pixel (PSNR)", color="white", fontsize=10)
    ax5.set_xticks(hs); ax5.set_xticklabels([f"t+{h}" for h in hs])
    ax5.legend(fontsize=7, labelcolor="white", facecolor="#222", edgecolor="#444")
    for h, p in zip(hs, psnrs):
        ax5.annotate(f"{p:.1f}", (h, p), textcoords="offset points",
                     xytext=(0, 8), ha="center", color="white", fontsize=8)

    # ── [1,2] Scorecard ─────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.set_facecolor("#1a1a1a"); ax6.axis("off")

    grade   = lambda r: "✓ Bon" if r > 0.8  else ("~ Moyen" if r > 0.5 else "✗ Faible")
    grade_u = lambda u: "✓ Bon" if u < -1.5 else ("~ Limite" if u < -0.5 else "✗ Collapse")
    grade_p = lambda p: "✓ Bon" if p > 28   else ("~ Moyen"  if p > 22  else "✗ Faible")

    lines = [
        ("SCORECARD (AE)",   "",                                              "white"),
        ("",                 "",                                              "white"),
        ("R² lin. global",   f"{r2_lin:.3f}  {grade(r2_lin)}",              "#a5d6a7"),
        ("R² MLP global",    f"{r2_mlp:.3f}  {grade(r2_mlp)}",              "#a5d6a7"),
        ("Uniformité",       f"{uniformity:.3f}  {grade_u(uniformity)}",     "#4fc3f7"),
        ("Alignement",       f"{alignment:.3f}",                             "#ff8a65"),
        ("PSNR recon.",      f"{psnr(rec_mse):.1f} dB  {grade_p(psnr(rec_mse))}",  "#ffcc80"),
        ("PSNR pred t+1",    f"{psnr(pred_mse):.1f} dB  {grade_p(psnr(pred_mse))}", "#ffcc80"),
        ("Pred. t+1 (z)",    f"{lat_sims.get(1, float('nan')):.3f}",         "#ce93d8"),
        ("Pred. t+10 (z)",   f"{lat_sims.get(10, float('nan')):.3f}",        "#ce93d8"),
    ]
    for i, (label, val, color) in enumerate(lines):
        ax6.text(0.05, 0.95 - i * 0.10, label, transform=ax6.transAxes,
                 color=color, fontsize=9,
                 fontweight="bold" if label.startswith("SCORECARD") else "normal")
        ax6.text(0.58, 0.95 - i * 0.10, val, transform=ax6.transAxes,
                 color="white", fontsize=9)

    fig.suptitle("Évaluation LeWorldModelRec (AE)", color="white",
                 fontsize=13, y=0.98)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=DARK)
        print(f"\nFigure sauvegardée : {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    device = get_device()
    print(f"Device : {device}")
    model = load_model(args.checkpoint, device)
    train_loader, val_loader = make_loaders(args.dataset_dir)
    print(f"Train : {len(train_loader.dataset)} traj  |  Val : {len(val_loader.dataset)} traj")

    r2s_lin, r2_lin, lin_preds, s_val, r2s_mlp, r2_mlp = linear_probe(
        model, train_loader, val_loader, device, n_epochs=args.probe_epochs)

    rec_mse, pred_mse = pixel_quality(model, val_loader, device)

    _, _, seqs_train = collect_embeddings(model, train_loader, device, normalize=True)
    _, _, seqs_val   = collect_embeddings(model, val_loader,   device, normalize=True)
    uniformity, alignment = uniformity_alignment(seqs_train, seqs_val)

    lat_sims, pix_mses = prediction_horizon(
        model, val_loader, device, horizons=args.horizons)

    print("\n── Résumé ────────────────────────────────────────────────")
    print(f"  R² global (linéaire) : {r2_lin:.4f}")
    print(f"  R² global (MLP)      : {r2_mlp:.4f}")
    print(f"  PSNR reconstruction  : {10*np.log10(1/(rec_mse+1e-10)):.1f} dB")
    print(f"  PSNR prédiction t+1  : {10*np.log10(1/(pred_mse+1e-10)):.1f} dB")
    print(f"  Uniformité           : {uniformity:.4f}")
    print(f"  Alignement           : {alignment:.4f}")

    save_figure(r2s_lin, r2_lin, lin_preds, s_val,
                r2s_mlp, r2_mlp,
                rec_mse, pred_mse,
                uniformity, alignment,
                lat_sims, pix_mses,
                args.save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",   required=True)
    parser.add_argument("--dataset-dir",  default="dataset/pendulum")
    parser.add_argument("--probe-epochs", type=int, default=200)
    parser.add_argument("--horizons",     type=int, nargs="+", default=[1, 2, 5, 10])
    parser.add_argument("--save",         default=None)
    main(parser.parse_args())
