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

  # Export GIF (même épisode que les autres visuels)
  python3 tools/dream_explorer.py --model jepa --traj-idx 124 --save-gif visuals/dream_explorer_jepa.gif
  python3 tools/dream_explorer.py --model rec  --traj-idx 124 --save-gif visuals/dream_explorer_ae.gif
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
import torch
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
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
    return model, None, "AE", C_REC


def decode_frame(z1d, model, decoder, model_type, device):
    """z (D,) → numpy (H, W, 3) dans [0,1]."""
    z = torch.from_numpy(z1d).float().to(device).unsqueeze(0)
    with torch.no_grad():
        if model_type == "JEPA":
            frame = decoder(z)
        else:
            frame = model.decode(z)
    return frame[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy()


# ── Précomputation ────────────────────────────────────────────────────────────

def precompute(model, decoder, model_type, dataset_dir, device,
               n_episodes, n_steps, seed_frames=2, seq_len=100,
               traj_idx=None):
    """
    Background PCA  : premiers n_episodes épisodes (0..n_episodes-1).
    Dream featured  : traj_idx si fourni, sinon tous les n_episodes épisodes.

    Retourne bg_pc, S_bg, dreams_z, dreams_pc, dreams_fr, Vt3, mu, var.
    Si traj_idx est fourni, dreams_* contient un seul rêve (cet épisode).
    """
    ds    = PendulumSeqDataset(dataset_dir, seq_len=max(seq_len, seed_frames + 2))
    n_bg  = min(n_episodes, len(ds))

    # ── Fond : PCA sur n_bg épisodes ──────────────────────────────────────────
    print(f"  Encodage fond ({n_bg} trajectoires)…")
    bg_z, bg_st = [], []
    with torch.no_grad():
        for i in range(n_bg):
            frames, states = ds[i]
            z = model.encode(frames.unsqueeze(0).to(device))
            bg_z.append(z[0].cpu().numpy())
            bg_st.append(states.numpy())

    Z_bg = np.concatenate(bg_z,  axis=0)
    S_bg = np.concatenate(bg_st, axis=0)

    mu    = Z_bg.mean(0)
    Zc    = Z_bg - mu
    n_fit = min(len(Zc), 12000)
    rng   = np.random.RandomState(0)
    idx   = rng.choice(len(Zc), n_fit, replace=False)
    _, sv, Vt = np.linalg.svd(Zc[idx], full_matrices=False)
    Vt3   = Vt[:3]
    var   = (sv[:3] ** 2) / (sv ** 2).sum()
    bg_pc = Zc @ Vt3.T
    print(f"  PCA  PC1={var[0]:.1%}  PC2={var[1]:.1%}  PC3={var[2]:.1%}")

    # ── Rêves ─────────────────────────────────────────────────────────────────
    dream_indices = [traj_idx] if traj_idx is not None else list(range(n_bg))
    n_dreams = len(dream_indices)
    print(f"  Génération rêve(s) ({n_dreams} épisode(s) × {n_steps} pas)…")

    dreams_z, dreams_pc, dreams_fr = [], [], []
    with torch.no_grad():
        for ep_i, ep_idx in enumerate(dream_indices):
            frames, _ = ds[ep_idx]
            seed   = frames[:seed_frames].unsqueeze(0).to(device)
            z_seed = model.encode(seed)
            z0     = z_seed[0, -1:]

            traj_z = [z0.cpu().numpy()[0]]
            z_cur  = z0.unsqueeze(0)
            for _ in range(n_steps):
                z_cur = model.predictor(z_cur)
                traj_z.append(z_cur[0, 0].cpu().numpy())

            traj_z  = np.array(traj_z)
            traj_pc = (traj_z - mu) @ Vt3.T
            traj_fr = np.stack([
                decode_frame(traj_z[t], model, decoder, model_type, device)
                for t in range(len(traj_z))
            ])

            dreams_z.append(traj_z)
            dreams_pc.append(traj_pc)
            dreams_fr.append(traj_fr)

            if (ep_i + 1) % 5 == 0 or n_dreams <= 5:
                print(f"    épisode {ep_i+1}/{n_dreams}")

    print("  Précomputation terminée.")
    return bg_pc, S_bg, dreams_z, dreams_pc, dreams_fr, Vt3, mu, var


# ── Probe linéaire ────────────────────────────────────────────────────────────

def fit_state_probe(Z_bg, S_bg):
    mu    = Z_bg.mean(0)
    sigma = Z_bg.std(0) + 1e-8
    Zn    = (Z_bg - mu) / sigma
    W, _, _, _ = np.linalg.lstsq(
        np.c_[Zn, np.ones(len(Zn))], S_bg, rcond=None
    )
    return mu, sigma, W


def predict_states(z_traj, probe_mu, probe_sigma, probe_W):
    Zn = (z_traj - probe_mu) / probe_sigma
    return np.c_[Zn, np.ones(len(Zn))] @ probe_W


# ── Export GIF ────────────────────────────────────────────────────────────────

def save_gif(bg_pc, S_bg, dream_pc, dream_z, dream_fr, var,
             model_color, model_name,
             probe_mu, probe_sigma, probe_W,
             out_path, fps=12, elev=22, azim=-55):
    """
    GIF animé : espace latent 3D (gauche) + frame décodée (droite).
    La trajectoire du rêve s'anime pas à pas avec une traîne colorée.
    """
    n_frames = len(dream_fr)
    norm_th  = Normalize(vmin=-np.pi, vmax=np.pi)
    cmap_th  = plt.cm.hsv
    states   = predict_states(dream_z, probe_mu, probe_sigma, probe_W)

    fig = plt.figure(figsize=(10, 4.5), facecolor=DARK)
    fig.patch.set_facecolor(DARK)

    gs = gridspec.GridSpec(
        1, 2, figure=fig,
        width_ratios=[1.5, 1], wspace=0.03,
        left=0.01, right=0.99, top=0.90, bottom=0.04,
    )
    ax3d  = fig.add_subplot(gs[0, 0], projection="3d")
    ax_im = fig.add_subplot(gs[0, 1])

    # ── Style 3D ──────────────────────────────────────────────────────────────
    ax3d.set_facecolor(DARK2)
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(GRID)
    ax3d.tick_params(colors=DIM, labelsize=5.5)
    ax3d.view_init(elev=elev, azim=azim)

    # Fond : trajectoires réelles (très transparentes)
    rng = np.random.RandomState(0)
    sub = rng.choice(len(bg_pc), min(len(bg_pc), 5000), replace=False)
    ax3d.scatter(bg_pc[sub, 0], bg_pc[sub, 1], bg_pc[sub, 2],
                 c=S_bg[sub, 0], cmap=cmap_th, norm=norm_th,
                 s=1.5, alpha=0.9, linewidths=0, depthshade=True)

    # Trajectoire complète du rêve (très atténuée)
    ax3d.plot(dream_pc[:, 0], dream_pc[:, 1], dream_pc[:, 2],
              color=model_color, lw=0.7, alpha=0.2)

    ax3d.set_xlabel(f"PC1 ({var[0]:.0%})", color=DIM, fontsize=5.5, labelpad=1)
    ax3d.set_ylabel(f"PC2 ({var[1]:.0%})", color=DIM, fontsize=5.5, labelpad=1)
    ax3d.set_zlabel(f"PC3 ({var[2]:.0%})", color=DIM, fontsize=5.5, labelpad=1)
    ax3d.set_title("Latent space  R³", color=WHITE, fontsize=8, pad=5)

    # Éléments animés
    trail_line, = ax3d.plot([], [], [], color=model_color, lw=2.0, alpha=0.9)
    dot = ax3d.scatter([], [], [], s=70, color="white",
                       edgecolors=model_color, linewidths=1.8, zorder=10)

    # ── Frame ─────────────────────────────────────────────────────────────────
    ax_im.set_facecolor(DARK)
    ax_im.axis("off")
    ax_im.set_title(f"{model_name} — dreaming", color=model_color, fontsize=9, pad=5)
    im_art = ax_im.imshow(dream_fr[0], interpolation="nearest", aspect="equal")

    title = fig.text(0.5, 0.95, "t = 0", ha="center", color=DIM,
                     fontsize=8, fontfamily="monospace")

    TRAIL = 25   # longueur de la traîne

    def update(t):
        ts = max(0, t - TRAIL)
        tp = dream_pc[ts:t+1]
        if len(tp) > 1:
            trail_line.set_data_3d(tp[:, 0], tp[:, 1], tp[:, 2])
        dot._offsets3d = ([dream_pc[t, 0]], [dream_pc[t, 1]], [dream_pc[t, 2]])
        im_art.set_data(dream_fr[t])
        th, om = states[t, 0], states[t, 1]
        title.set_text(f"t={t:3d}  θ={th:+.2f}  ω={om:+.2f}")
        return [trail_line, dot, im_art, title]

    anim = animation.FuncAnimation(fig, update, frames=n_frames,
                                   interval=int(1000 / fps), blit=False)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    anim.save(out_path, fps=fps, writer="pillow")
    plt.close(fig)
    print(f"GIF sauvegardé : {out_path}")


# ── Viewer interactif ────────────────────────────────────────────────────────

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

    from matplotlib.widgets import Slider, Button
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

    norm_th = Normalize(vmin=-np.pi, vmax=np.pi)
    cmap_th = plt.cm.hsv

    ax3d.set_facecolor(DARK2)
    for pane in [ax3d.xaxis.pane, ax3d.yaxis.pane, ax3d.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(GRID)
    ax3d.tick_params(colors=DIM, labelsize=6)

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

    dream_line,  = ax3d.plot([], [], [], color=model_color, lw=1.5, alpha=0.9)
    dream_dot    = ax3d.scatter([], [], [], color="white", s=55, zorder=10,
                                edgecolors=model_color, linewidths=1.5)
    dream_scatter = ax3d.scatter([], [], [], c=[], cmap=cmap_th, norm=norm_th,
                                 s=14, alpha=0.9, linewidths=0, depthshade=False)

    ax_img.set_facecolor(DARK2)
    ax_img.axis("off")
    im_artist = ax_img.imshow(dreams_fr[0][0], interpolation="bilinear", aspect="auto")
    title_img = ax_img.set_title("Frame imaginée  t=0", color=WHITE, fontsize=9)

    ax_st.set_facecolor(DARK2)
    for sp in ax_st.spines.values(): sp.set_edgecolor(GRID)
    ax_st.tick_params(colors=DIM, labelsize=7)
    ax_st.grid(True, color=GRID, lw=0.4, alpha=0.7)
    ax_st.set_xlabel("t (frames)", color=DIM, fontsize=7)

    t_ax    = np.arange(n_steps + 1)
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

    handles = [line_th, line_om]
    labels  = [l.get_label() for l in handles]
    ax_st.legend(handles, labels, facecolor="#1e1e1e", labelcolor=WHITE,
                 edgecolor=GRID, fontsize=7, loc="upper right")

    txt_states = fig.text(0.73, 0.915, "", color=WHITE, fontsize=8.5,
                          fontfamily="monospace")

    _state = {"ep": 0, "t": 0}

    def _refresh():
        ep = _state["ep"]
        t  = _state["t"]
        pc     = dreams_pc[ep]
        z_tr   = dreams_z[ep]
        fr     = dreams_fr[ep]
        states = predict_states(z_tr, probe_mu, probe_sigma, probe_W)

        dream_line.set_data_3d(pc[:, 0], pc[:, 1], pc[:, 2])
        dream_scatter._offsets3d = (pc[:, 0], pc[:, 1], pc[:, 2])
        dream_scatter.set_array(states[:, 0])
        dream_dot._offsets3d = ([pc[t, 0]], [pc[t, 1]], [pc[t, 2]])

        im_artist.set_data(fr[t])
        title_img.set_text(f"Frame imaginée   t={t}")

        line_th.set_ydata(states[:, 0])
        line_om.set_ydata(states[:, 1])
        ax_st.set_ylim(states[:, 0].min() - 0.3, states[:, 0].max() + 0.3)
        ax_st2.set_ylim(states[:, 1].min() - 0.5, states[:, 1].max() + 0.5)
        vline_th.set_xdata([t])
        vline_om.set_xdata([t])

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

    _play = {"running": False, "timer": None}

    def _tick():
        t_next = (_state["t"] + 1) % (n_steps + 1)
        _state["t"] = t_next
        sl_t.set_val(t_next)

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

    fig.text(0.04, 0.005,
             "← → sliders épisode / temps  |  clic+drag sur la figure 3D pour pivoter",
             color=DIM, fontsize=7)
    plt.show()


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    save_mode = bool(args.save_gif)
    if save_mode:
        matplotlib.use("Agg")
    else:
        matplotlib.use("TkAgg")

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

    traj_idx = args.traj_idx if args.traj_idx >= 0 else None

    (bg_pc, S_bg,
     dreams_z, dreams_pc, dreams_fr,
     Vt3, mu, var) = precompute(
        model, decoder, model_name,
        args.dataset_dir, device,
        n_episodes=args.n_episodes,
        n_steps=args.n_steps,
        seed_frames=2,
        seq_len=args.seq_len,
        traj_idx=traj_idx,
    )

    # Probe linéaire z → (θ, ω) fitté sur les z du fond
    print("\n  Fitting probe linéaire…")
    ds = PendulumSeqDataset(args.dataset_dir, seq_len=args.seq_len)
    raw_z, raw_s = [], []
    with torch.no_grad():
        for i in range(min(args.n_episodes, len(ds))):
            frames, states = ds[i]
            z = model.encode(frames.unsqueeze(0).to(device))[0].cpu().numpy()
            raw_z.append(z)
            raw_s.append(states.numpy())
    Z_raw = np.concatenate(raw_z, axis=0)
    S_raw = np.concatenate(raw_s, axis=0)
    probe_mu, probe_sigma, probe_W = fit_state_probe(Z_raw, S_raw)
    print("  Probe prêt.")

    del model
    if decoder is not None:
        del decoder

    if save_mode:
        save_gif(
            bg_pc, S_bg,
            dreams_pc[0], dreams_z[0], dreams_fr[0],
            var, model_color, model_name,
            probe_mu, probe_sigma, probe_W,
            out_path=args.save_gif,
            fps=args.fps,
        )
    else:
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
                   help="Trajectoires pour le fond PCA")
    p.add_argument("--traj-idx",     type=int, default=-1,
                   help="Épisode à rêver (-1 = tous les n_episodes)")
    p.add_argument("--n-steps",      type=int, default=100)
    p.add_argument("--seq-len",      type=int, default=100)
    p.add_argument("--save-gif",     default=None,
                   help="Chemin GIF export (désactive le viewer interactif)")
    p.add_argument("--fps",          type=int, default=12)
    main(p.parse_args())
