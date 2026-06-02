"""
Visualisation de l'espace latent projeté dans R³ (PCA 3D).

Chaque point = embedding z d'une frame.
Les frames consécutives d'une même trajectoire sont reliées par une courbe.
Couleur = θ (angle du pendule) via colormap circulaire HSV.

Mode interactif (défaut) : fenêtre matplotlib rotatable à la souris.
Mode sauvegarde (--save)  : grille 4 angles de vue → PNG.

Usage :
  # JEPA
  python3 tools/visualize_latent_3d.py --model jepa
  python3 tools/visualize_latent_3d.py --model jepa --color omega
  python3 tools/visualize_latent_3d.py --model jepa --save visuals/latent3d_jepa.png

  # AE
  python3 tools/visualize_latent_3d.py --model rec
  python3 tools/visualize_latent_3d.py --model rec  --save visuals/latent3d_rec.png

  # Comparaison côte à côte
  python3 tools/visualize_latent_3d.py --model both --save visuals/latent3d_compare.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

from data.dataset import PendulumSeqDataset

DARK  = "#111111"
DARK2 = "#1a1a1a"
GRID  = "#2a2a2a"
WHITE = "#e0e0e0"
C_DIM = "#666666"


# ── Chargement des modèles ────────────────────────────────────────────────────

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
    return model, f"JEPA  ep{ckpt.get('epoch','?')}"


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
    return model, f"AE  ep{ckpt.get('epoch','?')}"


# ── Extraction des embeddings ─────────────────────────────────────────────────

def extract(model, dataset_dir, device, n_trajs=40, seq_len=100):
    """
    Encode n_trajs trajectoires.
    Retourne :
      trajs_pc  : list[ (T, 3) ]   coordonnées PCA de chaque trajectoire
      trajs_st  : list[ (T, 2) ]   états (θ, ω)
      var_exp   : (3,)              variance expliquée par PC1/PC2/PC3
    """
    ds = PendulumSeqDataset(dataset_dir, seq_len=seq_len)
    n  = min(len(ds), n_trajs)

    all_z, traj_boundaries = [], [0]
    all_states = []

    with torch.no_grad():
        for i in range(n):
            frames, states = ds[i]                           # (T, 3, H, W), (T, 2)
            z = model.encode(frames.unsqueeze(0).to(device)) # (1, T, D)
            all_z.append(z[0].cpu().numpy())
            all_states.append(states.numpy())
            traj_boundaries.append(traj_boundaries[-1] + frames.shape[0])

    Z = np.concatenate(all_z, axis=0)     # (N, D)
    S = np.concatenate(all_states, axis=0) # (N, 2)

    # PCA 3D
    mu  = Z.mean(0)
    Zc  = Z - mu
    n_fit = min(len(Zc), 10000)
    rng   = np.random.RandomState(0)
    idx   = rng.choice(len(Zc), n_fit, replace=False)
    _, sv, Vt = np.linalg.svd(Zc[idx], full_matrices=False)
    PC    = Zc @ Vt[:3].T                  # (N, 3)
    var   = (sv[:3] ** 2) / (sv ** 2).sum()

    # Séparer par trajectoire
    trajs_pc, trajs_st = [], []
    for i in range(n):
        a, b = traj_boundaries[i], traj_boundaries[i + 1]
        trajs_pc.append(PC[a:b])
        trajs_st.append(S[a:b])

    return trajs_pc, trajs_st, var


# ── Encodage couleur état (θ, ω) → RGB ───────────────────────────────────────

def state_to_rgb(theta, omega, omega_95p=None):
    """
    θ  → Teinte  (circulaire : -π=rouge, 0=cyan, +π=rouge)
    |ω| → Luminosité  (0=sombre, max=vif)
    Saturation = 0.88 fixe.

    Retourne (rgb (N,3), omega_95p) — omega_95p peut être réutilisé pour
    la légende afin de garder la même normalisation.
    """
    import colorsys
    theta = np.asarray(theta, dtype=float)
    omega = np.asarray(omega, dtype=float)

    h = (theta + np.pi) / (2 * np.pi)          # [0, 1] circulaire
    if omega_95p is None:
        omega_95p = float(np.percentile(np.abs(omega), 95)) + 1e-8
    v = np.clip(np.abs(omega) / omega_95p, 0, 1)
    v = 0.25 + 0.75 * v                         # [0.25, 1.0] — jamais trop sombre

    rgb = np.array([colorsys.hsv_to_rgb(float(h[i]), 0.88, float(v[i]))
                    for i in range(len(h))], dtype=float)
    return rgb, omega_95p


def _state_legend_image(omega_95p, n_th=60, n_om=40):
    """Grille θ×ω → RGB pour la légende 2D."""
    import colorsys
    th_vals = np.linspace(-np.pi, np.pi, n_th)
    om_vals = np.linspace(-omega_95p, omega_95p, n_om)
    img = np.zeros((n_om, n_th, 3))
    for j, om in enumerate(om_vals):
        for i, th in enumerate(th_vals):
            h = (th + np.pi) / (2 * np.pi)
            v = 0.25 + 0.75 * min(abs(om) / omega_95p, 1)
            img[n_om - 1 - j, i] = colorsys.hsv_to_rgb(h, 0.88, v)
    return img, th_vals, om_vals


# ── Dessin 3D ─────────────────────────────────────────────────────────────────

def draw_latent_3d(ax, trajs_pc, trajs_st, var, title, color_by="theta",
                   n_trajs_curves=12, alpha_curve=0.6, alpha_scatter=0.35):
    """
    Remplit un Axes3D.

    color_by : "theta"  (angle, hsv)
               "omega"  (vitesse, coolwarm)
               "time"   (progression temporelle, plasma)
               "state"  (θ→teinte + |ω|→luminosité, espace d'état complet)
    """
    ax.set_facecolor(DARK2)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.set_edgecolor(GRID)

    ax.tick_params(colors=C_DIM, labelsize=6.5)
    ax.xaxis.label.set_color(C_DIM)
    ax.yaxis.label.set_color(C_DIM)
    ax.zaxis.label.set_color(C_DIM)

    # Normalisation couleur
    all_st  = np.concatenate(trajs_st, axis=0)
    all_pc  = np.concatenate(trajs_pc, axis=0)
    _omega_95p = None   # utilisé par le mode "state" pour la légende

    if color_by == "state":
        # Encodage complet : θ→teinte, |ω|→luminosité
        rgb_all, _omega_95p = state_to_rgb(all_st[:, 0], all_st[:, 1])
        cmap, norm = None, None
        cbar_label = "état (θ→teinte · |ω|→luminosité)"
    else:
        if color_by == "theta":
            cmap   = plt.cm.hsv
            c_vals = all_st[:, 0]
            vmin, vmax = -np.pi, np.pi
            cbar_label = "θ (rad)"
        elif color_by == "omega":
            cmap   = plt.cm.coolwarm
            c_vals = all_st[:, 1]
            p = np.percentile(np.abs(c_vals), 95)
            vmin, vmax = -p, p
            cbar_label = "ω (rad/s)"
        else:  # time
            cmap   = plt.cm.plasma
            lens   = [len(t) for t in trajs_pc]
            c_vals = np.concatenate([np.linspace(0, 1, l) for l in lens])
            vmin, vmax = 0, 1
            cbar_label = "t / T"
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        rgb_all = None

    # Scatter (tous les points, subsampled)
    if len(all_pc) > 5000:
        rng = np.random.RandomState(1)
        idx = rng.choice(len(all_pc), 5000, replace=False)
        sc_pts = all_pc[idx]
        sc_col = rgb_all[idx] if color_by == "state" else c_vals[idx]
    else:
        sc_pts = all_pc
        sc_col = rgb_all if color_by == "state" else c_vals

    if color_by == "state":
        sc = ax.scatter(sc_pts[:, 0], sc_pts[:, 1], sc_pts[:, 2],
                        c=sc_col, s=2, alpha=alpha_scatter,
                        linewidths=0, depthshade=True)
    else:
        sc = ax.scatter(sc_pts[:, 0], sc_pts[:, 1], sc_pts[:, 2],
                        c=sc_col, cmap=cmap, norm=norm,
                        s=2, alpha=alpha_scatter, linewidths=0, depthshade=True)

    # Courbes pour les N premières trajectoires
    for i, (pc, st) in enumerate(zip(trajs_pc[:n_trajs_curves], trajs_st[:n_trajs_curves])):
        if color_by == "state":
            rgb_traj, _ = state_to_rgb(st[:, 0], st[:, 1], _omega_95p)
            for t in range(len(pc) - 1):
                c = (rgb_traj[t] + rgb_traj[t + 1]) / 2
                ax.plot(pc[t:t+2, 0], pc[t:t+2, 1], pc[t:t+2, 2],
                        color=c, lw=0.9, alpha=alpha_curve)
            ax.scatter(*pc[0], color=rgb_traj[0],
                       s=18, marker="o", zorder=5, edgecolors="white", linewidths=0.3)
        else:
            if color_by == "theta":   seg_col = st[:, 0]
            elif color_by == "omega": seg_col = st[:, 1]
            else:                     seg_col = np.linspace(0, 1, len(pc))
            for t in range(len(pc) - 1):
                c = cmap(norm((seg_col[t] + seg_col[t+1]) / 2))
                ax.plot(pc[t:t+2, 0], pc[t:t+2, 1], pc[t:t+2, 2],
                        color=c, lw=0.9, alpha=alpha_curve)
            ax.scatter(*pc[0], color=cmap(norm(seg_col[0])),
                       s=18, marker="o", zorder=5, edgecolors="white", linewidths=0.3)

    ax.set_xlabel(f"PC1 ({var[0]:.1%})", fontsize=7, labelpad=2)
    ax.set_ylabel(f"PC2 ({var[1]:.1%})", fontsize=7, labelpad=2)
    ax.set_zlabel(f"PC3 ({var[2]:.1%})", fontsize=7, labelpad=2)
    ax.set_title(title, color=WHITE, fontsize=10, pad=8)

    return sc, norm, cmap, cbar_label, _omega_95p


def add_colorbar(fig, sc, norm, cmap, label, ax, omega_95p=None):
    if cmap is None:
        # Mode "state" — légende 2D θ×ω
        img, th_vals, om_vals = _state_legend_image(omega_95p)
        inset = ax.inset_axes([1.05, 0.15, 0.25, 0.5])
        inset.imshow(img, aspect="auto",
                     extent=[th_vals[0], th_vals[-1], om_vals[0], om_vals[-1]])
        inset.set_xlabel("θ (rad)", color=C_DIM, fontsize=6)
        inset.set_ylabel("ω (rad/s)", color=C_DIM, fontsize=6)
        inset.set_title("État\n(θ, ω)", color=C_DIM, fontsize=6)
        inset.tick_params(colors=C_DIM, labelsize=5.5)
        inset.set_xticks([-np.pi, 0, np.pi])
        inset.set_xticklabels(["-π", "0", "π"])
        for sp in inset.spines.values(): sp.set_edgecolor(GRID)
        return

    cbar = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.08, aspect=20)
    cbar.set_label(label, color=C_DIM, fontsize=7)
    cbar.ax.tick_params(colors=C_DIM, labelsize=6.5)
    cbar.outline.set_edgecolor(GRID)
    if label == "θ (rad)":
        cbar.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        cbar.set_ticklabels(["-π", "-π/2", "0", "π/2", "π"])


# ── Modes de visualisation ────────────────────────────────────────────────────

def single_interactive(trajs_pc, trajs_st, var, title, color_by):
    """Fenêtre interactive — rotatable à la souris."""
    fig = plt.figure(figsize=(10, 8), facecolor=DARK)
    ax  = fig.add_subplot(111, projection="3d", facecolor=DARK2)
    sc, norm, cmap, cbar_label, omega_95p = draw_latent_3d(
        ax, trajs_pc, trajs_st, var, title, color_by=color_by)
    add_colorbar(fig, sc, norm, cmap, cbar_label, ax, omega_95p=omega_95p)
    fig.suptitle(f"Espace latent R³ — {title}", color=WHITE, fontsize=12, y=0.98)
    plt.tight_layout()
    plt.show()


def four_views(fig, axes_row, trajs_pc, trajs_st, var, title, color_by):
    """
    4 vues fixes dans une ligne d'axes : face, dessus, côté, isométrique.
    """
    views = [
        (20, -60,  "Vue isométrique"),
        (90,  -90, "Vue de dessus (PC1-PC2)"),
        (0,   -90, "Vue de face (PC1-PC3)"),
        (0,    0,  "Vue de côté (PC2-PC3)"),
    ]
    sc_ref = norm_ref = cmap_ref = label_ref = omega_ref = None

    for ax, (elev, azim, view_title) in zip(axes_row, views):
        sc, norm, cmap, cbar_label, omega_95p = draw_latent_3d(
            ax, trajs_pc, trajs_st, var,
            title=f"{title}\n{view_title}",
            color_by=color_by,
            n_trajs_curves=8,
            alpha_curve=0.55,
            alpha_scatter=0.25,
        )
        ax.view_init(elev=elev, azim=azim)
        sc_ref, norm_ref, cmap_ref, label_ref, omega_ref = sc, norm, cmap, cbar_label, omega_95p

    return sc_ref, norm_ref, cmap_ref, label_ref, omega_ref


def save_figure_single(trajs_pc, trajs_st, var, title, color_by, out_path):
    """4 vues d'un seul modèle."""
    fig = plt.figure(figsize=(20, 5), facecolor=DARK)
    axes = [fig.add_subplot(1, 4, i+1, projection="3d") for i in range(4)]
    sc, norm, cmap, label, omega_95p = four_views(fig, axes, trajs_pc, trajs_st, var, title, color_by)
    add_colorbar(fig, sc, norm, cmap, label, axes[-1], omega_95p=omega_95p)
    fig.suptitle(f"Espace latent R³  —  {title}  [coloré par {label}]",
                 color=WHITE, fontsize=12, y=1.01)
    plt.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"Sauvegardé → {out_path}")


def save_figure_both(trajs_j, states_j, var_j,
                     trajs_r, states_r, var_r,
                     color_by, out_path):
    """2 × 4 vues : JEPA (ligne 1) et AE (ligne 2)."""
    fig = plt.figure(figsize=(20, 10), facecolor=DARK)
    gs  = plt.GridSpec(2, 4, figure=fig, hspace=0.15, wspace=0.05)

    axes_j = [fig.add_subplot(gs[0, i], projection="3d") for i in range(4)]
    axes_r = [fig.add_subplot(gs[1, i], projection="3d") for i in range(4)]

    sc_j, norm_j, cmap_j, label_j, omega_j = four_views(
        fig, axes_j, trajs_j, states_j, var_j, "JEPA", color_by)
    sc_r, norm_r, cmap_r, label_r, omega_r = four_views(
        fig, axes_r, trajs_r, states_r, var_r, "AE", color_by)

    add_colorbar(fig, sc_j, norm_j, cmap_j, label_j, axes_j[-1], omega_95p=omega_j)
    add_colorbar(fig, sc_r, norm_r, cmap_r, label_r, axes_r[-1], omega_95p=omega_r)

    fig.suptitle(f"Espace latent R³ — JEPA vs AE  [coloré par {label_j}]",
                 color=WHITE, fontsize=13, y=1.005)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=DARK)
    plt.close(fig)
    print(f"Sauvegardé → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    device = torch.device("mps") if torch.backends.mps.is_available() \
        else torch.device("cuda") if torch.cuda.is_available() \
        else torch.device("cpu")
    print(f"Device : {device}")

    if args.save:
        matplotlib.use("Agg")

    do_jepa = args.model in ("jepa", "both")
    do_rec  = args.model in ("rec",  "both")

    trajs_j = states_j = var_j = label_j = None
    trajs_r = states_r = var_r = label_r = None

    if do_jepa:
        print(f"Chargement JEPA : {args.jepa_ckpt}")
        model_j, label_j = load_jepa(args.jepa_ckpt, device)
        print(f"Encodage ({args.n_trajs} trajectoires × {args.seq_len} frames)…")
        trajs_j, states_j, var_j = extract(
            model_j, args.dataset_dir, device, args.n_trajs, args.seq_len)
        print(f"  Variance expliquée : PC1={var_j[0]:.1%}  PC2={var_j[1]:.1%}  PC3={var_j[2]:.1%}")
        del model_j

    if do_rec:
        print(f"Chargement AE : {args.rec_ckpt}")
        model_r, label_r = load_rec(args.rec_ckpt, device)
        print(f"Encodage ({args.n_trajs} trajectoires × {args.seq_len} frames)…")
        trajs_r, states_r, var_r = extract(
            model_r, args.dataset_dir, device, args.n_trajs, args.seq_len)
        print(f"  Variance expliquée : PC1={var_r[0]:.1%}  PC2={var_r[1]:.1%}  PC3={var_r[2]:.1%}")
        del model_r

    if args.save:
        if args.model == "both":
            save_figure_both(trajs_j, states_j, var_j,
                             trajs_r, states_r, var_r,
                             args.color, args.save)
        elif do_jepa:
            save_figure_single(trajs_j, states_j, var_j, label_j, args.color, args.save)
        else:
            save_figure_single(trajs_r, states_r, var_r, label_r, args.color, args.save)
    else:
        # Mode interactif — un modèle à la fois
        if do_jepa:
            single_interactive(trajs_j, states_j, var_j, label_j, args.color)
        if do_rec:
            single_interactive(trajs_r, states_r, var_r, label_r, args.color)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",       default="jepa",
                   choices=["jepa", "rec", "both"],
                   help="Modèle(s) à visualiser")
    p.add_argument("--color",       default="theta",
                   choices=["theta", "omega", "time", "state"],
                   help="Variable de couleur : theta | omega | time | state (θ+ω simultanés)")
    p.add_argument("--jepa-ckpt",   default="checkpoints/jepa/lewm_best.pt")
    p.add_argument("--rec-ckpt",    default="checkpoints/rec/lewm_rec_best.pt")
    p.add_argument("--dataset-dir", default="dataset/pendulum")
    p.add_argument("--n-trajs",     type=int, default=40,
                   help="Nombre de trajectoires à encoder")
    p.add_argument("--seq-len",     type=int, default=100,
                   help="Frames par trajectoire")
    p.add_argument("--save",        default=None,
                   help="Chemin PNG (mode non-interactif)")
    main(p.parse_args())
