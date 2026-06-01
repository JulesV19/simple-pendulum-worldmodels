"""
Comparaison en temps réel : frames réelles / reconstructions / frames imaginées.

LeWorldModelRec embarque son propre décodeur — pas de checkpoint séparé.

Trois panneaux :
  Réel          — frame originale
  Reconstruction — encode(frame_t) → decode  (ce que le modèle "voit")
  Imaginé        — encode(seed) → rollout predictor → decode

Contrôles :
  Espace / bouton Pause  — play / pause
  Slider                 — scrubbing
  < Prev / Next >        — changer de trajectoire

Usage:
  python3 imagine_rec.py
  python3 imagine_rec.py --n-steps 40 --traj-idx 0
  python3 imagine_rec.py --gif --n-steps 40
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider

from models.rec.model import LeWorldModelRec
from data.dataset import PendulumSeqDataset


DARK    = "#111"
DARK2   = "#1a1a1a"
C_REAL  = "#4fc3f7"
C_REC   = "#a5d6a7"
C_DREAM = "#ff8a65"

N_SEED = 2   # frames réelles pour initialiser z (diff ≠ 0 → ω dans z)


# ── Chargement ─────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def load_model(path: str, device) -> LeWorldModelRec:
    ckpt  = torch.load(path, map_location=device, weights_only=False)
    args  = ckpt.get("args", {})
    model = LeWorldModelRec(
        embed_dim       = args.get("embed_dim",       128),
        hidden_dim      = args.get("hidden_dim",      512),
        lam             = args.get("lam",             0.5),
        n_proj          = args.get("n_proj",          512),
        perceptual_coef = 0.0,   # pas de VGG en inférence
        freq_coef       = 0.0,
    ).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model


# ── Dreaming ───────────────────────────────────────────────────────────────────

def _decode_chunked(model: LeWorldModelRec, z: torch.Tensor, chunk: int = 16) -> np.ndarray:
    """
    Décode (1, T, D) en chunks pour éviter l'OOM sur MPS/CPU.
    Retourne (T, H, W, 3) uint8.
    """
    T = z.shape[1]
    frames = []
    for start in range(0, T, chunk):
        z_chunk = z[:, start:start + chunk]             # (1, chunk, D)
        out = model.decode(z_chunk)                     # (1, chunk, 3, H, W)
        frames.append(out[0].clamp(0, 1)
                             .permute(0, 2, 3, 1)
                             .cpu().numpy())
    arr = np.concatenate(frames, axis=0)                # (T, H, W, 3)
    return (arr * 255).astype(np.uint8)


@torch.no_grad()
def build_dream(model: LeWorldModelRec,
                frames_tensor: torch.Tensor,
                n_steps: int, device):
    """
    frames_tensor : (1, T, 3, 64, 64) normalisées [0, 1]

    Retourne :
      real_np  : (T, H, W, 3)          uint8  — frames originales
      rec_np   : (T, H, W, 3)          uint8  — reconstructions encode→decode
      dream_np : (n_steps+1, H, W, 3)  uint8  — imaginé depuis z seed
    """
    frames_tensor = frames_tensor.to(device)

    # Reconstruction : encode toute la séquence puis decode par chunks
    z_all  = model.encode(frames_tensor)                # (1, T, D)
    rec_np = _decode_chunked(model, z_all)

    # Dream : seules N_SEED frames données à l'encodeur → z0 shape (1, 1, D)
    seed   = frames_tensor[:, :N_SEED]                  # (1, N_SEED, 3, H, W)
    z_seed = model.encode(seed)                         # (1, N_SEED, D)
    z0     = z_seed[:, -1:]                             # (1, 1, D)
    z_traj = model.imagine(z0, n_steps)                 # (1, n_steps+1, D)
    dream_np = _decode_chunked(model, z_traj)

    # Frames originales
    real_np = (frames_tensor[0].permute(0, 2, 3, 1)
                                .cpu().numpy() * 255).astype(np.uint8)

    return real_np, rec_np, dream_np


# ── Viewer interactif ──────────────────────────────────────────────────────────

class DreamViewer:
    def __init__(self, model, dataset, args, device):
        self.model   = model
        self.dataset = dataset
        self.args    = args
        self.device  = device

        self.idx     = args.traj_idx if args.traj_idx >= 0 else random.randint(0, len(dataset) - 1)
        self.t       = 0
        self.playing = True

        self._load()
        self._build()
        self._start()

    # ── Chargement ──────────────────────────────────────────────────────────────

    def _load(self):
        frames, _ = self.dataset[self.idx]              # (T, 3, 64, 64)
        frames_t  = frames.unsqueeze(0)                 # (1, T, 3, 64, 64)
        n_steps   = min(self.args.n_steps, frames.shape[0] - N_SEED - 1)

        print(f"Trajectoire {self.idx}  —  dreaming {n_steps} steps…", end=" ", flush=True)
        self.real_np, self.rec_np, self.dream_np = build_dream(
            self.model, frames_t, n_steps, self.device)
        print("ok")

        self.real_start = N_SEED - 1
        self.T = min(len(self.real_np) - self.real_start, len(self.dream_np))
        self.t = 0

    # ── Figure ──────────────────────────────────────────────────────────────────

    def _build(self):
        self.fig = plt.figure(figsize=(14, 5.5), facecolor=DARK)
        self.fig.patch.set_facecolor(DARK)

        outer = gridspec.GridSpec(
            2, 1, figure=self.fig,
            height_ratios=[10, 1.2],
            hspace=0.12,
            left=0.03, right=0.97, top=0.91, bottom=0.06,
        )

        gs = gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=outer[0],
            wspace=0.07, width_ratios=[1, 1, 1, 0.55],
        )

        self.ax_real  = self.fig.add_subplot(gs[0, 0])
        self.ax_rec   = self.fig.add_subplot(gs[0, 1])
        self.ax_dream = self.fig.add_subplot(gs[0, 2])
        self.ax_info  = self.fig.add_subplot(gs[0, 3])

        for ax, title, col in [
            (self.ax_real,  "Réel",                   C_REAL),
            (self.ax_rec,   "Reconstruction",          C_REC),
            (self.ax_dream, "Imaginé (z rollout)",     C_DREAM),
        ]:
            ax.set_facecolor(DARK)
            ax.axis("off")
            ax.set_title(title, color=col, fontsize=11, pad=6)
            for sp in ax.spines.values():
                sp.set_edgecolor(col)
                sp.set_linewidth(1.5)
                sp.set_visible(True)

        self.ax_info.set_facecolor(DARK2)
        self.ax_info.axis("off")

        # Contrôles
        ctrl = gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=outer[1],
            wspace=0.25, width_ratios=[1, 1, 1, 0.2, 5],
        )
        ax_prev  = self.fig.add_subplot(ctrl[0, 0])
        ax_play  = self.fig.add_subplot(ctrl[0, 1])
        ax_next  = self.fig.add_subplot(ctrl[0, 2])
        ax_slide = self.fig.add_subplot(ctrl[0, 4])

        self.btn_prev = Button(ax_prev, "<  Prev", color="#222", hovercolor="#444")
        self.btn_play = Button(ax_play, "Pause",   color="#222", hovercolor="#444")
        self.btn_next = Button(ax_next, "Next  >", color="#222", hovercolor="#444")
        self.slider   = Slider(ax_slide, "Frame", 0, max(self.T - 1, 1),
                               valinit=0, valstep=1, color=C_REAL)

        for btn in (self.btn_prev, self.btn_play, self.btn_next):
            btn.label.set_color("white")
            btn.label.set_fontsize(9)
        self.slider.label.set_color("white")
        self.slider.valtext.set_color("white")

        self.btn_prev.on_clicked(self._prev)
        self.btn_play.on_clicked(self._toggle)
        self.btn_next.on_clicked(self._next)
        self.slider.on_changed(self._on_slide)

        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        self.im_real  = self.ax_real.imshow(blank,  interpolation="nearest")
        self.im_rec   = self.ax_rec.imshow(blank,   interpolation="nearest")
        self.im_dream = self.ax_dream.imshow(blank, interpolation="nearest")

        self._update_title()
        self._update_info()
        self._draw(0)

    # ── Affichage ───────────────────────────────────────────────────────────────

    def _draw(self, t):
        real_idx = self.real_start + t
        self.im_real.set_data(self.real_np[min(real_idx,  len(self.real_np)  - 1)])
        self.im_rec.set_data(self.rec_np[min(real_idx,    len(self.rec_np)   - 1)])
        self.im_dream.set_data(self.dream_np[min(t,       len(self.dream_np) - 1)])
        self.slider.eventson = False
        self.slider.set_val(t)
        self.slider.eventson = True

    def _update_title(self):
        n = len(self.dataset)
        self.fig.suptitle(
            f"Dream viewer  —  trajectoire {self.idx + 1} / {n}  "
            f"(seed : {N_SEED} frames réelles)",
            color="white", fontsize=11, y=0.97,
        )

    def _update_info(self):
        ax = self.ax_info
        ax.clear(); ax.axis("off"); ax.set_facecolor(DARK2)

        real_t  = min(self.real_start + self.t, len(self.real_np) - 1)
        dream_t = min(self.t, len(self.dream_np) - 1)

        lines = [
            ("DREAM INFO",    "",                         "white"),
            ("",              "",                         "white"),
            ("t réel",        str(real_t),                C_REAL),
            ("t imaginé",     str(dream_t),               C_DREAM),
            ("",              "",                         "white"),
            ("Seed frames",   str(N_SEED),                "#aaa"),
            ("Steps totaux",  str(self.T - 1),            "#aaa"),
            ("",              "",                         "white"),
            ("dt",            "0.05 s",                   "#aaa"),
            ("Durée rêvée",   f"{(self.T-1)*0.05:.2f} s", "#aaa"),
        ]
        for i, (label, val, color) in enumerate(lines):
            y = 0.97 - i * 0.09
            w = "bold" if label == "DREAM INFO" else "normal"
            ax.text(0.05, y, label, transform=ax.transAxes,
                    color=color, fontsize=8, fontweight=w, va="top")
            ax.text(0.6,  y, val,   transform=ax.transAxes,
                    color="white", fontsize=8, va="top")

    # ── Animation ───────────────────────────────────────────────────────────────

    def _animate(self, _):
        if self.playing:
            self.t = (self.t + 1) % self.T
            self._draw(self.t)
            if self.t % 5 == 0:
                self._update_info()
        return [self.im_real, self.im_rec, self.im_dream]

    def _start(self):
        interval = max(50, int(1000 / self.args.fps))
        self.anim = animation.FuncAnimation(
            self.fig, self._animate,
            interval=interval, blit=True, cache_frame_data=False,
        )
        plt.show()

    # ── Callbacks ───────────────────────────────────────────────────────────────

    def _toggle(self, _):
        self.playing = not self.playing
        self.btn_play.label.set_text("Pause" if self.playing else "Play")
        self.fig.canvas.draw_idle()

    def _on_slide(self, val):
        self.t = int(val)
        self._draw(self.t)
        self._update_info()

    def _prev(self, _):
        self.idx = (self.idx - 1) % len(self.dataset)
        self._reload()

    def _next(self, _):
        self.idx = (self.idx + 1) % len(self.dataset)
        self._reload()

    def _reload(self):
        self._load()
        self.slider.valmax = max(self.T - 1, 1)
        self.slider.ax.set_xlim(0, max(self.T - 1, 1))
        self._update_title()
        self._update_info()
        self._draw(0)
        self.fig.canvas.draw_idle()


# ── Mode GIF ───────────────────────────────────────────────────────────────────

def save_gif(real_np, rec_np, dream_np, real_start, path, fps):
    T = min(len(real_np) - real_start, len(dream_np))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), facecolor=DARK)
    fig.patch.set_facecolor(DARK)

    for ax, title, col in zip(axes,
                               ["Réel", "Reconstruction", "Imaginé"],
                               [C_REAL, C_REC, C_DREAM]):
        ax.set_facecolor(DARK); ax.axis("off")
        ax.set_title(title, color=col, fontsize=12)

    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    im_r = axes[0].imshow(blank, interpolation="nearest")
    im_c = axes[1].imshow(blank, interpolation="nearest")
    im_d = axes[2].imshow(blank, interpolation="nearest")
    txt  = fig.text(0.5, 0.02, "t = 0", ha="center", color="#999", fontsize=9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    def update(t):
        idx = min(real_start + t, len(real_np) - 1)
        im_r.set_data(real_np[idx])
        im_c.set_data(rec_np[idx])
        im_d.set_data(dream_np[min(t, len(dream_np) - 1)])
        txt.set_text(f"t = {t}  ({t * 0.05:.2f} s)")
        return [im_r, im_c, im_d, txt]

    anim = animation.FuncAnimation(fig, update, frames=T,
                                   interval=int(1000 / fps), blit=True)
    anim.save(path, fps=fps, writer="pillow")
    plt.close(fig)
    print(f"GIF sauvegardé : {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    device = get_device()
    print(f"Device : {device}")

    model   = load_model(args.checkpoint, device)
    dataset = PendulumSeqDataset(args.dataset_dir)

    if args.gif:
        idx = args.traj_idx if args.traj_idx >= 0 else random.randint(0, len(dataset) - 1)
        frames, _ = dataset[idx]
        n_steps   = min(args.n_steps, frames.shape[0] - N_SEED - 1)
        real_np, rec_np, dream_np = build_dream(
            model, frames.unsqueeze(0), n_steps, device)
        Path(args.vis_dir).mkdir(parents=True, exist_ok=True)
        gif_path = f"{args.vis_dir}/dream_rec_{idx:04d}.gif"
        save_gif(real_np, rec_np, dream_np, N_SEED - 1, gif_path, args.fps)
    else:
        DreamViewer(model, dataset, args, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  default="checkpoints/rec/lewm_rec_best.pt")
    parser.add_argument("--dataset-dir", default="dataset/pendulum")
    parser.add_argument("--n-steps",     type=int, default=100)
    parser.add_argument("--traj-idx",    type=int, default=-1)
    parser.add_argument("--fps",         type=int, default=12)
    parser.add_argument("--gif",         action="store_true")
    parser.add_argument("--vis-dir",     default="visuals")
    main(parser.parse_args())
