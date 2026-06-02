"""
Dream Explorer — visualisation interactive de l'espace latent R³ + rêve.

Pour chaque épisode :
  • Le modèle encode les 2 premières frames réelles → z_0
  • Le predictor roule n_steps pas → z_1, z_2, …  (le "rêve")
  • Chaque z_t est décodé → frame imaginée
  • La trajectoire du rêve est projetée dans R³ (même PCA que le fond réel)

Fenêtre interactive :
  Gauche     : espace latent 3D (fond = réel, coloré θ | avant-plan = rêve courant)
  Haut droit : frame imaginée à l'instant t
  Bas droit  : courbes θ et ω du rêve (marqueur à t)
  Sliders    : épisode  /  instant t

Usage :
  python3 tools/dream_explorer.py --model jepa
  python3 tools/dream_explorer.py --model rec
  python3 tools/dream_explorer.py --model jepa --n-episodes 30 --n-steps 120
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("TkAgg")          # backend interactif
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401

from data.dataset import PendulumSeqDataset

DARK  = "#0e0e0e"
DARK2 = "#161616"
GRID  = "#282828"
WHITE = "#e4e4e4"
DIM   = "#555555"
C_JEPA = "#4fc3f7"
C_REC  = "#ff8a65"


# ── Chargement ────────────────────────────────────────────────────────────────

def load_jepa(ckpt_path, decoder_path, device):
    from models.jepa.model import LeWorldModel
    from models.decoder    import Decoder

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

    embed_dim = a.get("embed_dim", 128)
    decoder = Decoder(embed_dim=embed_dim).to(device)
    dckpt   = torch.load(decoder_path, map_location=device, weights_only=False)
    decoder.load_state_dict(dckpt["decoder"])
    decoder.eval()
    for p in decoder.parameters(): p.requires_grad_(False)

    epoch = ckpt.get("epoch", "?")
    print(f"  JEPA chargé  epoch={epoch}  embed_dim={embed_dim}")
    return model, decoder, "JEPA", C_JEPA


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

    epoch = ckpt.get("epoch", "?")
    print(f"  AE chargé  epoch={epoch}  embed_dim={a.get('embed_dim',128)}")
    # Pour REC, le décodeur est intégré — on expose une interface commune
    return model, None, "AE", C_REC


def decode_frame(z1d, model, decoder, model_type, device):
    """z (D,) → numpy (H, W, 3) dans [0,1]."""
    z = torch.from_numpy(z1d).float().to(device).unsqueeze(0)  # (1, D)
    with torch.no_grad():
        if model_type == "JEPA":
            frame = decoder(z)           # (1, 3, H, W)
        else:
            frame = model.decode(z)      # (1, 3, H, W)
    return frame[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()


# ── Précomputation ────────────────────────────────────────────────────────────

def precompute(model, decoder, model_type, dataset_dir, device,
               n_episodes, n_steps, seed_frames=2, seq_len=100):
    """
    Pour chaque épisode :
      - Encode seed_frames frames réelles → z_0
      - Rollout n_steps pas → z_dream (n_steps+1, D)
      - Decode chaque z_t → frame (H, W, 3)
    Retourne aussi les embeddings de fond (trajectoires réelles pour la PCA).
    """
    print(f"\n  Précomputation {n_episodes} épisodes × {n_steps} pas…")
    ds = PendulumSeqDataset(dataset_dir, seq_len=max(seq_len, seed_frames + 2))
    n_ep = min(n_episodes, len(ds))

    # ── Fond : encoder n_episodes trajectoires réelles pour la PCA ────────────
    bg_z, bg_st = [], []
    with torch.no_grad():
        for i in range(n_ep):
            frames, states = ds[i]
            z = model.encode(frames.unsqueeze(0).to(device))   # (1, T, D)
            bg_z.append(z[0].cpu().numpy())
            bg_st.append(states.numpy())

    Z_bg = np.concatenate(bg_z,  axis=0)   # (N, D)
    S_bg = np.concatenate(bg_st, axis=0)   # (N, 2)

    # PCA 3D sur le fond
    mu   = Z_bg.mean(0)
    Zc   = Z_bg - mu
    n_fit = min(len(Zc), 12000)
    rng   = np.random.RandomState(0)
    idx   = rng.choice(len(Zc), n_fit, replace=False)
    _, sv, Vt = np.linalg.svd(Zc[idx], full_matrices=False)
    Vt3   = Vt[:3]                         # (3, D) — axes PCA
    var   = (sv[:3] ** 2) / (sv ** 2).sum()

    bg_pc = Zc @ Vt3.T                     # (N, 3)
    print(f"  PCA  PC1={var[0]:.1%}  PC2={var[1]:.1%}  PC3={var[2]:.1%}")

    # ── Rêves ─────────────────────────────────────────────────────────────────
    dreams_z  = []   # list[ (n_steps+1, D) ]
    dreams_pc = []   # list[ (n_steps+1, 3) ]
    dreams_fr = []   # list[ (n_steps+1, H, W, 3) ]

    with torch.no_grad():
        for i in range(n_ep):
            frames, _ = ds[i]
            # Encoder les seed_frames premières frames pour avoir ω dans z_0
            seed = frames[:seed_frames].unsqueeze(0).to(device)   # (1, S, 3, H, W)
            z_seed = model.encode(seed)                           # (1, S, D)
            z0 = z_seed[0, -1:]                                   # (1, D) — dernier z encodé

            # Rollout
            traj_z = [z0.cpu().numpy()[0]]
            z_cur  = z0.unsqueeze(0)      # (1, 1, D)
            for _ in range(n_steps):
                z_cur = model.predictor(z_cur)
                traj_z.append(z_cur[0, 0].cpu().numpy())

            traj_z  = np.array(traj_z)    # (n_steps+1, D)
            traj_pc = (traj_z - mu) @ Vt3.T  # projeter dans le même espace PCA

            # Décoder
            traj_fr = np.stack([
                decode_frame(traj_z[t], model, decoder, model_type, device)
                for t in range(len(traj_z))
            ])

            dreams_z.append(traj_z)
            dreams_pc.append(traj_pc)
            dreams_fr.append(traj_fr)

            if (i + 1) % 5 == 0:
                print(f"    épisode {i+1}/{n_ep}")

    print("  Précomputation terminée.")
    return (bg_pc, S_bg,
            dreams_z, dreams_pc, dreams_fr,
            Vt3, mu, var)


# ── Estimation des états via probe linéaire ───────────────────────────────────

def fit_state_probe(Z_bg, S_bg):
    """lstsq z → (θ, ω) pour annoter le rêve."""
    mu    = Z_bg.mean(0)
    sigma = Z_bg.std(0) + 1e-8
    Zn    = (Z_bg - mu) / sigma
    W, _, _, _ = np.linalg.lstsq(
        np.c_[Zn, np.ones(len(Zn))], S_bg, rcond=None
    )
    return mu, sigma, W


def predict_states(z_traj, probe_mu, probe_sigma, probe_W):
    """z_traj (T, D) → states_pred (T, 2)."""
    Zn = (z_traj - probe_mu) / probe_sigma
    return np.c_[Zn, np.ones(len(Zn))] @ probe_W


# ── Figure interactive ────────────────────────────────────────────────────────

def run_explorer(bg_pc, S_bg,
                 dreams_z, dreams_pc, dreams_fr,
                 var, model_color, model_name,
                 probe_mu, probe_sigma, probe_W,
                 n_steps):

    n_ep = len(dreams_pc)
    fig  = plt.figure(figsize=(14, 8), facecolor=DARK)
    fig.canvas.manager.set_window_title(f"Dream Explorer — {model_name}")

    gs = gridspec.GridSpec(
        3, 2,
        figure=fig,
        width_ratios=[1.6, 1],
        height_ratios=[1, 0.55, 0.12],
        hspace=0.08, wspace=0.08,
        left=0.04, right=0.97, top=0.93, bottom=0.13,
    )

    ax3d   = fig.add_subplot(gs[0:2, 0], projection="3d")
    ax_img = fig.add_subplot(gs[0, 1])
    ax_st  = fig.add_subplot(gs[1, 1])

    # Sliders + bouton Play/Pause
    ax_sl_ep = fig.add_axes([0.10, 0.06, 0.74, 0.025], facecolor="#1e1e1e")
    ax_sl_t  = fig.add_axes([0.10, 0.02, 0.74, 0.025], facecolor="#1e1e1e")
    ax_btn   = fig.add_axes([0.87, 0.02, 0.10, 0.055], facecolor="#1e1e1e")

    sl_ep = Slider(ax_sl_ep, "Épisode", 0, n_ep - 1, valinit=0, valstep=1,
                   color=model_color)
    sl_t  = Slider(ax_sl_t,  "t",       0, n_steps,  valinit=0, valstep=1,
                   color=model_color)
    btn   = Button(ax_btn, "▶  Play", color="#1e1e1e", hovercolor="#2a2a2a")
    btn.label.set_color(WHITE)
    btn.label.set_fontsize(9)

    for sl in (sl_ep, sl_t):
        sl.label.set_color(WHITE)
        sl.valtext.set_color(WHITE)

    fig.suptitle(f"Dream Explorer — {model_name}  |  espace latent R³ + rêve",
                 color=WHITE, fontsize=11, y=0.97)

    # ── 3D fond ────────────────────────────────────────────────────────────────
    ax3d.set_facecolor(DARK2)
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(GRID)
    ax3d.tick_params(colors=DIM, labelsize=6)

    # Fond : trajectoires réelles colorées par θ, très transparentes
    norm_th = Normalize(vmin=-np.pi, vmax=np.pi)
    cmap_th = plt.cm.hsv

    rng = np.random.RandomState(0)
    sub = rng.choice(len(bg_pc), min(len(bg_pc), 6000), replace=False)
    ax3d.scatter(bg_pc[sub, 0], bg_pc[sub, 1], bg_pc[sub, 2],
                 c=S_bg[sub, 0], cmap=cmap_th, norm=norm_th,
                 s=1.5, alpha=0.18, linewidths=0, depthshade=True)

    ax3d.set_xlabel(f"PC1 ({var[0]:.1%})", color=DIM, fontsize=6, labelpad=1)
    ax3d.set_ylabel(f"PC2 ({var[1]:.1%})", color=DIM, fontsize=6, labelpad=1)
    ax3d.set_zlabel(f"PC3 ({var[2]:.1%})", color=DIM, fontsize=6, labelpad=1)
    ax3d.set_title("Espace latent  (fond=réel · courbe=rêve · coloré par θ)",
                   color=WHITE, fontsize=8, pad=6)

    # Éléments dynamiques (rêve)
    dream_line,  = ax3d.plot([], [], [], color=model_color, lw=1.5, alpha=0.9)
    dream_dot    = ax3d.scatter([], [], [], color="white", s=55, zorder=10,
                                edgecolors=model_color, linewidths=1.5)
    dream_scatter = ax3d.scatter([], [], [], c=[], cmap=cmap_th, norm=norm_th,
                                 s=14, alpha=0.9, linewidths=0, depthshade=False)

    # ── Image ──────────────────────────────────────────────────────────────────
    ax_img.set_facecolor(DARK2)
    ax_img.axis("off")
    H, W = dreams_fr[0][0].shape[:2]
    im_artist = ax_img.imshow(dreams_fr[0][0], interpolation="bilinear",
                               aspect="auto")
    title_img = ax_img.set_title("Frame imaginée  t=0", color=WHITE, fontsize=9)

    # ── Courbes θ / ω ──────────────────────────────────────────────────────────
    ax_st.set_facecolor(DARK2)
    for sp in ax_st.spines.values(): sp.set_edgecolor(GRID)
    ax_st.tick_params(colors=DIM, labelsize=7)
    ax_st.grid(True, color=GRID, lw=0.4, alpha=0.7)
    ax_st.set_xlabel("t (frames)", color=DIM, fontsize=7)

    t_ax = np.arange(n_steps + 1)
    states0 = predict_states(dreams_z[0], probe_mu, probe_sigma, probe_W)

    line_th, = ax_st.plot(t_ax, states0[:, 0], color="#4fc3f7", lw=1.3, label="θ estimé")
    ax_st2   = ax_st.twinx()
    ax_st2.set_facecolor(DARK2)
    ax_st2.tick_params(colors=DIM, labelsize=7)
    for sp in ax_st2.spines.values(): sp.set_edgecolor(GRID)
    line_om, = ax_st2.plot(t_ax, states0[:, 1], color="#ff8a65", lw=1.3, label="ω estimé")
    ax_st.set_ylabel("θ (rad)", color="#4fc3f7", fontsize=7)
    ax_st2.set_ylabel("ω (rad/s)", color="#ff8a65", fontsize=7)

    vline_th = ax_st.axvline(0, color="white", lw=1, ls="--", alpha=0.6)
    vline_om = ax_st2.axvline(0, color="white", lw=1, ls="--", alpha=0.6)

    # Légende combinée
    handles = [line_th, line_om]
    labels  = [l.get_label() for l in handles]
    ax_st.legend(handles, labels, facecolor="#1e1e1e", labelcolor=WHITE,
                 edgecolor=GRID, fontsize=7, loc="upper right")

    # Annotation valeurs courantes
    txt_states = fig.text(0.73, 0.915, "", color=WHITE, fontsize=8.5,
                          fontfamily="monospace")

    # ── Mise à jour ────────────────────────────────────────────────────────────

    _state = {"ep": 0, "t": 0}

    def _refresh():
        ep = _state["ep"]
        t  = _state["t"]

        pc    = dreams_pc[ep]        # (n_steps+1, 3)
        z_tr  = dreams_z[ep]         # (n_steps+1, D)
        fr    = dreams_fr[ep]        # (n_steps+1, H, W, 3)
        states = predict_states(z_tr, probe_mu, probe_sigma, probe_W)

        # Couleur θ pour les points du rêve
        theta_colors = states[:, 0]

        # 3D — trajectoire rêve
        dream_line.set_data_3d(pc[:, 0], pc[:, 1], pc[:, 2])
        dream_scatter._offsets3d = (pc[:, 0], pc[:, 1], pc[:, 2])
        dream_scatter.set_array(theta_colors)
        dream_dot._offsets3d = ([pc[t, 0]], [pc[t, 1]], [pc[t, 2]])

        # Image
        im_artist.set_data(fr[t])
        title_img.set_text(f"Frame imaginée   t={t}")

        # Courbes
        line_th.set_ydata(states[:, 0])
        line_om.set_ydata(states[:, 1])
        ax_st.set_ylim(states[:, 0].min() - 0.3, states[:, 0].max() + 0.3)
        ax_st2.set_ylim(states[:, 1].min() - 0.5, states[:, 1].max() + 0.5)
        vline_th.set_xdata([t])
        vline_om.set_xdata([t])

        # Annotation
        th_val = states[t, 0]
        om_val = states[t, 1]
        txt_states.set_text(f"θ = {th_val:+.3f} rad\nω = {om_val:+.3f} rad/s")

        fig.canvas.draw_idle()

    def on_episode(val):
        _state["ep"] = int(val)
        _state["t"]  = 0
        sl_t.set_val(0)
        _refresh()

    def on_time(val):
        _state["t"] = int(val)
        _refresh()

    sl_ep.on_changed(on_episode)
    sl_t.on_changed(on_time)

    # ── Play / Pause ───────────────────────────────────────────────────────────
    _play = {"running": False, "timer": None}

    def _tick():
        t_next = (_state["t"] + 1) % (n_steps + 1)
        _state["t"] = t_next
        sl_t.set_val(t_next)   # déclenche on_time → _refresh

    def on_play(event):
        if _play["running"]:
            _play["timer"].stop()
            _play["running"] = False
            btn.label.set_text("▶  Play")
        else:
            _play["timer"] = fig.canvas.new_timer(interval=80)
            _play["timer"].add_callback(_tick)
            _play["timer"].start()
            _play["running"] = True
            btn.label.set_text("⏸  Pause")
        fig.canvas.draw_idle()

    btn.on_clicked(on_play)

    _refresh()

    # Instructions
    fig.text(0.04, 0.005,
             "← → sliders épisode / temps  |  clic+drag sur la figure 3D pour pivoter",
             color=DIM, fontsize=7)

    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    device = (torch.device("mps")  if torch.backends.mps.is_available()  else
              torch.device("cuda") if torch.cuda.is_available()           else
              torch.device("cpu"))
    print(f"Device : {device}")

    if args.model == "jepa":
        model, decoder, model_name, model_color = load_jepa(
            args.jepa_ckpt, args.decoder_ckpt, device)
    else:
        model, decoder, model_name, model_color = load_rec(
            args.rec_ckpt, device)

    (bg_pc, S_bg,
     dreams_z, dreams_pc, dreams_fr,
     Vt3, mu, var) = precompute(
        model, decoder, model_name,
        args.dataset_dir, device,
        n_episodes=args.n_episodes,
        n_steps=args.n_steps,
        seed_frames=2,
        seq_len=args.seq_len,
    )

    # Toutes les trajectoires de fond concaténées pour le probe
    all_bg_z = np.concatenate([
        model.encode(
            torch.from_numpy(
                np.load(f)[None].astype(np.float32)
            ).to(device)
        ).squeeze(0).cpu().numpy()
        for f in []    # probe fitted on bg_pc back-projected
    ], axis=0) if False else None

    # Probe fitté sur les z de fond (back-projeter depuis PCA n'est pas utile,
    # on refit sur les z bruts stockés dans dreams_z + bg encodings)
    # On réutilise les z réels encodés pour fitter le probe
    print("\n  Fitting probe linéaire z → (θ, ω)…")
    # Re-encoder le fond pour avoir les z bruts
    ds = PendulumSeqDataset(args.dataset_dir, seq_len=args.seq_len)
    raw_z_list, raw_s_list = [], []
    with torch.no_grad():
        for i in range(min(args.n_episodes, len(ds))):
            frames, states = ds[i]
            z = model.encode(frames.unsqueeze(0).to(device))[0].cpu().numpy()
            raw_z_list.append(z)
            raw_s_list.append(states.numpy())
    Z_raw = np.concatenate(raw_z_list, axis=0)
    S_raw = np.concatenate(raw_s_list, axis=0)
    probe_mu, probe_sigma, probe_W = fit_state_probe(Z_raw, S_raw)
    print("  Probe prêt.")

    del model
    if decoder is not None:
        del decoder

    run_explorer(
        bg_pc, S_bg,
        dreams_z, dreams_pc, dreams_fr,
        var, model_color, model_name,
        probe_mu, probe_sigma, probe_W,
        n_steps=args.n_steps,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dream Explorer")
    p.add_argument("--model",        default="jepa", choices=["jepa", "rec"])
    p.add_argument("--jepa-ckpt",    default="checkpoints/jepa/lewm_best.pt")
    p.add_argument("--decoder-ckpt", default="checkpoints/jepa/decoder_best.pt")
    p.add_argument("--rec-ckpt",     default="checkpoints/rec/lewm_rec_best.pt")
    p.add_argument("--dataset-dir",  default="dataset/pendulum")
    p.add_argument("--n-episodes",   type=int, default=20,
                   help="Nombre d'épisodes préchargés")
    p.add_argument("--n-steps",      type=int, default=100,
                   help="Longueur du rêve (pas predictor)")
    p.add_argument("--seq-len",      type=int, default=100,
                   help="Fenêtre d'encodage des trajectoires de fond")
    main(p.parse_args())
