#!/usr/bin/env python3
"""
Dashboard de comparaison : JEPA vs AE (reconstruction pixel).

Charge les JSON de stats d'entraînement, les checkpoints (optionnel),
calcule le probe linéaire z → (θ, ω) et génère le dashboard complet.

Usage :
  python3 eval/dashboard.py
  python3 eval/dashboard.py --out visuals/dashboard.png
  python3 eval/dashboard.py --no-probe   # JSON only, skip model loading
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

# ── Palette ───────────────────────────────────────────────────────────────────
DARK   = "#111111"
DARK2  = "#1a1a1a"
GRID   = "#2a2a2a"
C_JEPA = "#4fc3f7"
C_REC  = "#ff8a65"
C_GOOD = "#a5d6a7"
C_DIM  = "#777777"
WHITE  = "#e8e8e8"

# ── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_JEPA_JSON = "visuals/training_stats_jepa.json"
DEFAULT_REC_JSON  = "visuals/training_stats_rec.json"
DEFAULT_JEPA_CKPT = "checkpoints/jepa/lewm_best.pt"
DEFAULT_REC_CKPT  = "checkpoints/rec/lewm_rec_best.pt"
DEFAULT_DATASET   = "dataset/pendulum"
DEFAULT_OUT       = "visuals/dashboard.png"


# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


# ── Chargement JSON ───────────────────────────────────────────────────────────

def load_json(path):
    p = Path(path)
    if not p.exists():
        print(f"  [warn] JSON introuvable : {path}")
        return None
    with open(p) as f:
        return json.load(f)


# ── Probe linéaire z → (θ, ω) ────────────────────────────────────────────────

def _simple_kde(data, xs, bw=2.5):
    """KDE gaussien sans dépendance externe."""
    d = np.asarray(data, dtype=float)
    return np.array([
        np.mean(np.exp(-0.5 * ((x - d) / bw) ** 2) / (bw * np.sqrt(2 * np.pi)))
        for x in xs
    ])


def _r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum(0)
    ss_tot = ((y_true - y_true.mean(0)) ** 2).sum(0)
    return 1.0 - ss_res / (ss_tot + 1e-10)


def extract_encodings(encode_fn, dataset_dir, device, seq_len=50, max_trajs=300):
    """
    Encode les trajectoires du dataset via encode_fn.
    Retourne Z (N, D) et Y (N, 2) = (θ, ω).
    """
    from data.dataset import PendulumSeqDataset
    ds = PendulumSeqDataset(dataset_dir, seq_len=seq_len)
    n  = min(len(ds), max_trajs)

    all_z, all_y = [], []
    with torch.no_grad():
        for i in range(n):
            frames, states = ds[i]                          # (T, 3, H, W), (T, 2)
            z = encode_fn(frames.unsqueeze(0).to(device))  # (1, T, D)
            all_z.append(z[0].cpu().numpy())
            all_y.append(states.numpy())

    return np.concatenate(all_z, 0), np.concatenate(all_y, 0)


def linear_probe(Z, Y, val_frac=0.2, seed=42):
    """Régression linéaire numpy. Retourne R²(θ), R²(ω) sur val set."""
    rng   = np.random.RandomState(seed)
    idx   = rng.permutation(len(Z))
    n_val = int(len(Z) * val_frac)
    tr, vl = idx[n_val:], idx[:n_val]

    mu, sigma = Z[tr].mean(0), Z[tr].std(0) + 1e-8
    Zn = (Z - mu) / sigma

    Zaug = np.c_[Zn[tr], np.ones(len(tr))]
    W, _, _, _ = np.linalg.lstsq(Zaug, Y[tr], rcond=None)

    Y_pred = np.c_[Zn[vl], np.ones(len(vl))] @ W
    r2     = _r2(Y[vl], Y_pred)
    return float(r2[0]), float(r2[1])


def pca2d(Z, n_fit=8000, seed=0):
    """PCA 2D. Retourne PC (N,2) et variance expliquée (2,)."""
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(Z), min(len(Z), n_fit), replace=False)
    mu  = Z.mean(0)
    _, s, Vt = np.linalg.svd(Z[idx] - mu, full_matrices=False)
    PC  = (Z - mu) @ Vt[:2].T
    var = (s[:2] ** 2) / (s ** 2).sum()
    return PC, var


def load_jepa(ckpt_path, device):
    from models.jepa.model import LeWorldModel
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    for p in model.parameters(): p.requires_grad_(False)
    return model


def load_rec(ckpt_path, device):
    from models.rec.model import LeWorldModelRec
    ckpt  = torch.load(ckpt_path, map_location=device, weights_only=False)
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
    for p in model.parameters(): p.requires_grad_(False)
    return model


def compute_probe_and_pca(args, device):
    """Charge les deux modèles et calcule probe + PCA. Retourne un dict."""
    out = {}
    dataset_ok = Path(args.dataset_dir).exists()
    if not dataset_ok:
        print(f"  [warn] Dataset introuvable : {args.dataset_dir}")
        return out

    for name, ckpt_path, loader in [
        ("jepa", args.jepa_ckpt, load_jepa),
        ("rec",  args.rec_ckpt,  load_rec),
    ]:
        if not Path(ckpt_path).exists():
            print(f"  [warn] Checkpoint {name.upper()} introuvable : {ckpt_path}")
            continue
        print(f"  Chargement {name.upper()}…")
        model = loader(ckpt_path, device)
        print(f"  Encodage {name.upper()} ({args.max_trajs} trajectoires)…")
        Z, Y = extract_encodings(model.encode, args.dataset_dir, device,
                                 seq_len=50, max_trajs=args.max_trajs)
        print(f"    {len(Z):,} frames — probe…")
        r2_th, r2_om = linear_probe(Z, Y)
        print(f"    R²(θ)={r2_th:.3f}  R²(ω)={r2_om:.3f}")
        PC, var = pca2d(Z)
        out[f"r2_{name}"]  = (r2_th, r2_om)
        out[f"pca_{name}"] = (PC, Y, var)
        del model

    return out


# ── Helpers matplotlib ────────────────────────────────────────────────────────

def style_ax(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(DARK2)
    ax.tick_params(colors=C_DIM, labelsize=7.5)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.8)
    if title:  ax.set_title(title, color=WHITE, fontsize=9, pad=4, fontweight="medium")
    if xlabel: ax.set_xlabel(xlabel, color=C_DIM, fontsize=7.5)
    if ylabel: ax.set_ylabel(ylabel, color=C_DIM, fontsize=7.5)
    ax.tick_params(axis="x", colors=C_DIM)
    ax.tick_params(axis="y", colors=C_DIM)


def mklegend(ax, **kw):
    ax.legend(facecolor="#1e1e1e", labelcolor=WHITE, edgecolor=GRID, fontsize=7, **kw)


# ── Panels ────────────────────────────────────────────────────────────────────

def panel_loss(ax, jdata, color, title):
    """Courbes train/val loss pour un modèle."""
    if jdata is None:
        style_ax(ax, title=title)
        ax.text(0.5, 0.5, "données manquantes", ha="center", va="center",
                color=C_DIM, transform=ax.transAxes)
        return
    cv   = jdata.get("convergence", {})
    tr   = cv.get("train_loss", [])
    vl   = cv.get("val_loss",   [])
    best = cv.get("best_epoch", None)
    ep   = range(1, len(tr) + 1)
    if tr: ax.plot(ep, tr, color=color, lw=1.5, label="train",  alpha=0.9)
    if vl: ax.plot(ep, vl, color=color, lw=1.5, label="val",    alpha=0.6, ls="--")
    if best and tr:
        ax.axvline(best, color=color, lw=0.8, ls=":", alpha=0.5)
        ax.text(best + 0.3, ax.get_ylim()[1] * 0.97, f"↓ep{best}",
                color=color, fontsize=6.5, va="top")
    style_ax(ax, title=title, xlabel="epoch", ylabel="loss")
    mklegend(ax)


def panel_sigreg(ax, j_jepa, j_rec):
    """SIGReg des deux modèles sur le même axe."""
    for jdata, color, label in [(j_jepa, C_JEPA, "JEPA"), (j_rec, C_REC, "AE")]:
        if jdata is None: continue
        sig = jdata.get("convergence", {}).get("sigreg", [])
        if sig:
            ax.plot(range(1, len(sig)+1), sig, color=color, lw=1.5, label=label)
    style_ax(ax, title="SIGReg — régularisation espace latent",
             xlabel="epoch", ylabel="SIGReg loss")
    mklegend(ax)
    # annotation: AE has higher SIGReg → harder to regularize
    ax.text(0.97, 0.95, "AE plus haut → espace latent\nplus difficile à régulariser",
            transform=ax.transAxes, ha="right", va="top",
            color=C_DIM, fontsize=6.5, linespacing=1.4)


def panel_pred_loss(ax, j_jepa, j_rec):
    """Pred loss en axes séparés (espaces incomparables)."""
    ax2 = ax.twinx()
    ax2.set_facecolor(DARK2)

    for sp in ax.spines.values():  sp.set_edgecolor(GRID)
    for sp in ax2.spines.values(): sp.set_edgecolor(GRID)
    ax.set_facecolor(DARK2)
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.8)

    lines = []
    if j_jepa:
        pred_j = j_jepa.get("convergence", {}).get("pred_loss", [])
        if pred_j:
            l, = ax.plot(range(1, len(pred_j)+1), pred_j, color=C_JEPA, lw=1.5)
            lines.append((l, "JEPA (cosine+MSE)"))
    if j_rec:
        pred_r = j_rec.get("convergence", {}).get("pred_loss", [])
        if pred_r:
            l, = ax2.plot(range(1, len(pred_r)+1), pred_r, color=C_REC, lw=1.5)
            lines.append((l, "AE (MSE pixel)"))

    ax.set_title("Pred loss — axes séparés (espaces incomparables)",
                 color=WHITE, fontsize=9, pad=4)
    ax.set_xlabel("epoch", color=C_DIM, fontsize=7.5)
    ax.set_ylabel("JEPA pred loss", color=C_JEPA, fontsize=7.5)
    ax2.set_ylabel("AE pred loss",  color=C_REC,  fontsize=7.5)
    ax.tick_params(colors=C_DIM, labelsize=7.5)
    ax2.tick_params(colors=C_DIM, labelsize=7.5)
    ax.yaxis.label.set_color(C_JEPA)
    ax2.yaxis.label.set_color(C_REC)

    if lines:
        handles, labels = zip(*lines)
        ax.legend(handles, labels, facecolor="#1e1e1e", labelcolor=WHITE,
                  edgecolor=GRID, fontsize=7)


def panel_r2(ax, r2_jepa, r2_rec):
    """Grouped bar chart R²(θ) et R²(ω)."""
    x = np.array([0.0, 1.0])
    w = 0.33
    th_j, om_j = r2_jepa if r2_jepa else (0.0, 0.0)
    th_r, om_r = r2_rec  if r2_rec  else (0.0, 0.0)

    ax.bar(x - w/2, [th_j, om_j], w, color=C_JEPA, alpha=0.85, label="JEPA")
    ax.bar(x + w/2, [th_r, om_r], w, color=C_REC,  alpha=0.85, label="AE")

    for vals, color, sign in [([th_j, om_j], C_JEPA, -1), ([th_r, om_r], C_REC, +1)]:
        for xi, v in zip(x, vals):
            if v > 0.01:
                ax.text(xi + sign * w/2, v + 0.008, f"{v:.3f}",
                        ha="center", va="bottom", color=color, fontsize=7, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["R²(θ)\nposition", "R²(ω)\nvitesse angulaire"], color=WHITE, fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.tick_params(axis="x", colors=WHITE, labelsize=8)
    # Highlight R²(ω) — the key metric
    ax.axvspan(0.58, 1.42, alpha=0.06, color=C_GOOD, zorder=0)
    ax.text(1, 1.02, "← métrique clé", ha="center", color=C_GOOD, fontsize=7)

    style_ax(ax, title="Probe linéaire z → (θ, ω)", ylabel="R²")
    mklegend(ax, loc="lower right")

    if r2_jepa is None and r2_rec is None:
        ax.text(0.5, 0.5, "Lancer sans --no-probe\npour calculer le probe",
                ha="center", va="center", color=C_DIM, fontsize=8,
                transform=ax.transAxes)


def panel_power_kde(ax, j_jepa, j_rec):
    """KDE de la distribution de puissance GPU."""
    plotted = False
    for jdata, color, label in [(j_jepa, C_JEPA, "JEPA"), (j_rec, C_REC, "AE")]:
        if jdata is None: continue
        readings = jdata.get("energy", {}).get("power_readings_W", [])
        if not readings: continue
        p = np.array(readings, dtype=float)
        mu = p.mean()
        xs = np.linspace(p.min() - 8, p.max() + 8, 300)
        kde = _simple_kde(p, xs, bw=max(p.std() * 0.5, 1.5))
        ax.fill_between(xs, kde, alpha=0.2, color=color)
        ax.plot(xs, kde, color=color, lw=1.5, label=f"{label}  μ={mu:.1f} W")
        ax.axvline(mu, color=color, lw=1, ls="--", alpha=0.6)
        plotted = True

    style_ax(ax, title="Distribution puissance GPU", xlabel="Puissance (W)", ylabel="densité")
    if plotted:
        mklegend(ax)
    else:
        ax.text(0.5, 0.5, "Pas de données power", ha="center", va="center",
                color=C_DIM, transform=ax.transAxes)


def panel_table(ax, j_jepa, j_rec, r2_jepa, r2_rec):
    """Tableau récapitulatif complet."""
    ax.axis("off")
    ax.set_facecolor(DARK2)
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)

    def g(jdata, *keys, fmt=None, default="—"):
        if jdata is None: return default
        v = jdata
        for k in keys:
            if isinstance(v, dict) and k in v:
                v = v[k]
            else:
                return default
        if fmt is None: return str(v)
        try: return fmt.format(v)
        except Exception: return str(v)

    r2j_th = f"{r2_jepa[0]:.3f}" if r2_jepa else "—"
    r2j_om = f"{r2_jepa[1]:.3f}" if r2_jepa else "—"
    r2r_th = f"{r2_rec[0]:.3f}"  if r2_rec  else "—"
    r2r_om = f"{r2_rec[1]:.3f}"  if r2_rec  else "—"

    # Efficiency : R²(ω) / Wh
    def eff(r2, jdata):
        if r2 is None or jdata is None: return "—"
        wh = jdata.get("energy", {}).get("total_Wh", 0)
        return f"{r2[1]/wh:.4f}" if wh > 0 else "—"

    rows = [
        # (label, jepa_val, rec_val, highlight)
        ("Paramètres (M)",      g(j_jepa, "memory", "model_params_M", fmt="{:.3f}"),    g(j_rec, "memory", "model_params_M", fmt="{:.3f}"),   False),
        ("Batch size",          g(j_jepa, "hyperparams", "batch_size"),                  g(j_rec, "hyperparams", "batch_size"),                 False),
        ("rollout_k",           g(j_jepa, "dataset", "rollout_k"),                       g(j_rec, "dataset", "rollout_k"),                      False),
        ("seq_len",             g(j_jepa, "dataset", "seq_len"),                         g(j_rec, "dataset", "seq_len"),                        False),
        ("",                    "",                                                        "",                                                    False),
        ("Durée (min)",         g(j_jepa, "time",   "total_min",   fmt="{:.1f}"),         g(j_rec, "time",   "total_min",   fmt="{:.1f}"),       False),
        ("Énergie (Wh)",        g(j_jepa, "energy", "total_Wh",    fmt="{:.2f}"),         g(j_rec, "energy", "total_Wh",    fmt="{:.2f}"),       False),
        ("Puissance moy (W)",   g(j_jepa, "energy", "avg_power_W", fmt="{:.1f}"),         g(j_rec, "energy", "avg_power_W", fmt="{:.1f}"),       False),
        ("Pic GPU (W)",         g(j_jepa, "energy", "avg_power_W", fmt="{:.1f}"),         g(j_rec, "energy", "avg_power_W", fmt="{:.1f}"),       False),
        ("Grad steps",          g(j_jepa, "time",   "gradient_steps"),                    g(j_rec, "time",   "gradient_steps"),                  False),
        ("",                    "",                                                        "",                                                    False),
        ("Best val loss",       g(j_jepa, "convergence", "best_val_loss", fmt="{:.5f}"),  g(j_rec, "convergence", "best_val_loss", fmt="{:.5f}"), False),
        ("Best epoch",          g(j_jepa, "convergence", "best_epoch"),                   g(j_rec, "convergence", "best_epoch"),                  False),
        ("",                    "",                                                        "",                                                    False),
        ("R²(θ)",               r2j_th,                                                   r2r_th,                                                False),
        ("R²(ω)  ← clé",        r2j_om,                                                   r2r_om,                                                True),
        ("R²(ω) / Wh",          eff(r2_jepa, j_jepa),                                     eff(r2_rec, j_rec),                                    True),
    ]

    header = ("Métrique", "JEPA", "AE")
    all_rows = [header] + [(r[0], r[1], r[2]) for r in rows]
    highlights = [False] + [r[3] for r in rows]

    n   = len(all_rows)
    xs  = [0.02, 0.50, 0.76]
    row_h = 1.0 / n

    for i, (row, hl) in enumerate(zip(all_rows, highlights)):
        y = 1.0 - (i + 0.5) * row_h

        # Background stripes
        bg = DARK if i % 2 == 0 else "#161616"
        if i == 0: bg = "#1f2937"
        if hl:     bg = "#0d2118"
        ax.axhspan(1 - (i+1)*row_h, 1 - i*row_h, color=bg, zorder=0)

        colors  = [WHITE, C_JEPA, C_REC] if i > 0 else [WHITE, WHITE, WHITE]
        weights = ["bold"] * 3 if i == 0 else ["normal", "bold", "bold"]
        for xi, val, col, wt in zip(xs, row, colors, weights):
            ax.text(xi, y, str(val), transform=ax.transAxes,
                    fontsize=8, color=col, fontweight=wt,
                    va="center", fontfamily="monospace")

        if 0 < i < n - 1:
            ax.axhline(1 - (i+1)*row_h, color=GRID, lw=0.4, zorder=1)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title("Résumé comparatif", color=WHITE, fontsize=9, pad=6)


def panel_pca(ax, pca_data, title):
    """
    PCA de l'espace latent.
    Points colorés par θ (angle) — révèle si la géométrie circulaire du pendule
    est encodée dans z. Un anneau bien formé = représentation riche de l'état.
    """
    style_ax(ax, title=title, xlabel="PC1", ylabel="PC2")

    if pca_data is None:
        ax.text(0.5, 0.5, "Checkpoint non disponible\n(lancer sans --no-probe)",
                ha="center", va="center", color=C_DIM, fontsize=9,
                transform=ax.transAxes)
        return

    PC, Y, var = pca_data
    theta = Y[:, 0]   # angle : structure torique attendue

    # Subsample pour performance
    if len(PC) > 6000:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(PC), 6000, replace=False)
        PC, theta = PC[idx], theta[idx]

    sc = ax.scatter(PC[:, 0], PC[:, 1], c=theta, cmap="hsv",
                    vmin=-np.pi, vmax=np.pi,
                    s=1.5, alpha=0.5, rasterized=True, linewidths=0)

    cbar = plt.colorbar(sc, ax=ax, shrink=0.65, pad=0.01, aspect=25)
    cbar.ax.tick_params(colors=C_DIM, labelsize=7)
    cbar.set_label("θ (rad)", color=C_DIM, fontsize=7)
    cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])
    cbar.outline.set_edgecolor(GRID)

    ax.set_title(f"{title}  [PC1 {var[0]:.1%} · PC2 {var[1]:.1%}]",
                 color=WHITE, fontsize=9, pad=4)
    ax.set_xlabel(f"PC1 ({var[0]:.1%})", color=C_DIM, fontsize=7.5)
    ax.set_ylabel(f"PC2 ({var[1]:.1%})", color=C_DIM, fontsize=7.5)

    # Annotation : anneau = bonne représentation angulaire
    ax.text(0.02, 0.98, "anneau = θ encodé dans z",
            transform=ax.transAxes, ha="left", va="top",
            color=C_DIM, fontsize=6.5)


# ── Dashboard ─────────────────────────────────────────────────────────────────

def build_dashboard(args):
    print("── Chargement JSON ──")
    j_jepa = load_json(args.jepa_json)
    j_rec  = load_json(args.rec_json)

    r2_jepa = r2_rec = None
    pca_jepa = pca_rec = None

    if not args.no_probe:
        print("── Probe & PCA ──")
        device = get_device()
        print(f"  Device : {device}")
        data = compute_probe_and_pca(args, device)
        r2_jepa  = data.get("r2_jepa")
        r2_rec   = data.get("r2_rec")
        pca_jepa = data.get("pca_jepa")
        pca_rec  = data.get("pca_rec")

    print("── Construction figure ──")

    fig = plt.figure(figsize=(22, 15), facecolor=DARK)
    fig.suptitle("JEPA vs AE — Dashboard comparatif",
                 color=WHITE, fontsize=14, fontweight="bold", y=0.985)

    gs = gridspec.GridSpec(
        3, 4,
        figure=fig,
        hspace=0.45, wspace=0.36,
        height_ratios=[1.0, 1.0, 1.4],
        left=0.05, right=0.97, top=0.96, bottom=0.04,
    )

    # ── Row 0 ──────────────────────────────────────────────────────────────────
    ax_jloss = fig.add_subplot(gs[0, 0])
    ax_rloss = fig.add_subplot(gs[0, 1])
    ax_sig   = fig.add_subplot(gs[0, 2])
    ax_table = fig.add_subplot(gs[0:2, 3])   # spans rows 0 + 1

    # ── Row 1 ──────────────────────────────────────────────────────────────────
    ax_r2    = fig.add_subplot(gs[1, 0])
    ax_pred  = fig.add_subplot(gs[1, 1])
    ax_power = fig.add_subplot(gs[1, 2])
    # ax_table already spans [0:2, 3]

    # ── Row 2 ──────────────────────────────────────────────────────────────────
    ax_pca_j = fig.add_subplot(gs[2, 0:2])
    ax_pca_r = fig.add_subplot(gs[2, 2:4])

    # ── Fill ───────────────────────────────────────────────────────────────────
    panel_loss(ax_jloss, j_jepa, C_JEPA, "JEPA — Convergence (train / val)")
    panel_loss(ax_rloss, j_rec,  C_REC,  "AE   — Convergence (train / val)")
    panel_sigreg(ax_sig, j_jepa, j_rec)
    panel_table(ax_table, j_jepa, j_rec, r2_jepa, r2_rec)

    panel_r2(ax_r2, r2_jepa, r2_rec)
    panel_pred_loss(ax_pred, j_jepa, j_rec)
    panel_power_kde(ax_power, j_jepa, j_rec)

    panel_pca(ax_pca_j, pca_jepa, "Espace latent JEPA — PCA  (coloré par θ)")
    panel_pca(ax_pca_r, pca_rec,  "Espace latent AE   — PCA  (coloré par θ)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"\n  Dashboard → {out}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dashboard JEPA vs AE")
    p.add_argument("--jepa-json",   default=DEFAULT_JEPA_JSON)
    p.add_argument("--rec-json",    default=DEFAULT_REC_JSON)
    p.add_argument("--jepa-ckpt",   default=DEFAULT_JEPA_CKPT)
    p.add_argument("--rec-ckpt",    default=DEFAULT_REC_CKPT)
    p.add_argument("--dataset-dir", default=DEFAULT_DATASET)
    p.add_argument("--out",         default=DEFAULT_OUT)
    p.add_argument("--max-trajs",   type=int, default=300,
                   help="Nombre de trajectoires pour le probe et la PCA")
    p.add_argument("--no-probe",    action="store_true",
                   help="Passe le chargement des modèles — JSON uniquement")
    args = p.parse_args()
    build_dashboard(args)
