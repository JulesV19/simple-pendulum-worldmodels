"""
Comparaison côte-à-côte : LeWorldModel (JEPA) vs LeWorldModelRec (AE).

Affiche sur la même fenêtre :
  Réel | JEPA imaginé | AE imaginé | Stats

Panel Stats :
  - Coût d'inférence (temps + énergie estimée) pour chaque phase
  - Précision probe linéaire z → (θ, ω)

Usage :
  python3 compare_imagine.py
  python3 compare_imagine.py --n-steps 80 --probe-trajs 20
  python3 compare_imagine.py --gif --n-steps 60
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Button, Slider
from torch.utils.data import random_split

from models.jepa.model import LeWorldModel
from models.decoder  import Decoder
from models.rec.model import LeWorldModelRec
from data.dataset import PendulumSeqDataset


# ── Palette ───────────────────────────────────────────────────────────────────

DARK    = "#111"
DARK2   = "#1a1a1a"
C_REAL  = "#4fc3f7"
C_JEPA  = "#ff8a65"
C_AE    = "#a5d6a7"
C_DIM   = "#888"

N_SEED  = 2
CACHE   = Path("visuals/probe_cache.json")


# ── Device & énergie ──────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")


def device_tdp_watts(device: torch.device) -> float | None:
    """
    Retourne la puissance estimée du device en watts.
    CUDA : tente nvidia-smi pour la valeur réelle.
    MPS  : estimation conservatrice Apple M-series GPU.
    CPU  : estimation.
    """
    if device.type == "cuda":
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=power.draw",
                 "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, text=True,
            ).strip().split("\n")[0]
            return float(out)
        except Exception:
            return 15.0   # fallback
    if device.type == "mps":
        return 10.0       # Apple M-series GPU ~7-15 W
    return 10.0           # CPU fallback


def ms_to_mj(ms: float, watts: float) -> float:
    """Énergie en millijoules depuis un temps en ms et une puissance en W."""
    return watts * ms / 1000.0


# ── Chargement des modèles ────────────────────────────────────────────────────

def load_jepa(path: str, device):
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
    return model


def load_jepa_decoder(path: str, embed_dim: int, device):
    ckpt    = torch.load(path, map_location=device, weights_only=False)
    decoder = Decoder(embed_dim=embed_dim).to(device)
    decoder.load_state_dict(ckpt["decoder"])
    decoder.eval()
    return decoder


def load_rec(path: str, device):
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
    return model


# ── Benchmark d'inférence ────────────────────────────────────────────────────

def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def benchmark_inference(jepa_model, jepa_decoder, rec_model,
                        seed_frames: torch.Tensor, n_steps: int,
                        device, n_warmup: int = 3, n_runs: int = 10) -> dict:
    """
    Mesure le temps médian de chaque phase pour JEPA et AE.
    seed_frames : (1, N_SEED, 3, H, W)

    Phases mesurées :
      encode  — encoder.encode(seed)
      rollout — model.imagine(z0, n_steps)
      decode  — decoder(z_traj)  (chunked pour AE)
    """

    def time_fn(fn):
        # Warm-up
        for _ in range(n_warmup):
            fn()
        # Mesures
        times = []
        for _ in range(n_runs):
            _sync(device)
            t0 = time.perf_counter()
            fn()
            _sync(device)
            times.append((time.perf_counter() - t0) * 1000)  # ms
        return float(np.median(times))

    results = {}

    with torch.no_grad():

        # ── JEPA ──────────────────────────────────────────────────────────────
        z_seed_j = jepa_model.encode(seed_frames)
        z0_j     = z_seed_j[:, -1:]

        results["jepa_encode_ms"]  = time_fn(lambda: jepa_model.encode(seed_frames))
        results["jepa_rollout_ms"] = time_fn(lambda: jepa_model.imagine(z0_j, n_steps))

        z_traj_j = jepa_model.imagine(z0_j, n_steps)   # (1, n_steps+1, D)
        results["jepa_decode_ms"]  = time_fn(lambda: jepa_decoder(z_traj_j[0]))

        # ── AE ────────────────────────────────────────────────────────────────
        z_seed_r = rec_model.encode(seed_frames)
        z0_r     = z_seed_r[:, -1:]

        results["rec_encode_ms"]  = time_fn(lambda: rec_model.encode(seed_frames))
        results["rec_rollout_ms"] = time_fn(lambda: rec_model.imagine(z0_r, n_steps))

        z_traj_r = rec_model.imagine(z0_r, n_steps)    # (1, n_steps+1, D)
        # Decode chunked (même logique que imagine_rec.py)
        def _decode_rec():
            out = []
            for s in range(0, z_traj_r.shape[1], 16):
                out.append(rec_model.decode(z_traj_r[:, s:s+16]))
            return out
        results["rec_decode_ms"] = time_fn(_decode_rec)

    for prefix in ("jepa", "rec"):
        results[f"{prefix}_total_ms"] = (
            results[f"{prefix}_encode_ms"]
            + results[f"{prefix}_rollout_ms"]
            + results[f"{prefix}_decode_ms"]
        )

    return results


# ── Quick probe linéaire ──────────────────────────────────────────────────────

def r2_score(pred, true):
    ss_res = ((true - pred) ** 2).sum(0)
    ss_tot = ((true - true.mean(0)) ** 2).sum(0).clamp(min=1e-8)
    return (1.0 - ss_res / ss_tot).tolist()


@torch.no_grad()
def _encode_trajs(model, traj_indices, dataset_dir, device, chunk=16):
    ds = PendulumSeqDataset(dataset_dir, seq_len=None)
    Zs, Ls = [], []
    for i in traj_indices:
        frames, states = ds[i]
        T = frames.shape[0]
        z_chunks = []
        for s in range(0, T, chunk):
            fc = frames[s:s+chunk].unsqueeze(0).to(device)
            z_chunks.append(model.encode(fc)[0].cpu())
        z = torch.cat(z_chunks, dim=0)
        Zs.append(z[1:]); Ls.append(states[1:])
    return torch.cat(Zs).float(), torch.cat(Ls).float()


def quick_probe(jepa_model, rec_model, dataset_dir, n_trajs, device, seed=42):
    """
    Probe linéaire rapide sur n_trajs trajectoires.
    Retourne un dict avec R²(θ) et R²(ω) pour chaque modèle.
    Résultat mis en cache dans CACHE pour ne pas recalculer à chaque lancement.
    """
    if CACHE.exists():
        try:
            cached = json.loads(CACHE.read_text())
            if cached.get("n_trajs") == n_trajs:
                print(f"Probe chargé depuis cache ({CACHE})")
                return cached
        except Exception:
            pass

    print(f"Calcul du probe rapide ({n_trajs} trajectoires)…")
    ds  = PendulumSeqDataset(dataset_dir, seq_len=None)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), min(n_trajs, len(ds)), replace=False).tolist()
    n_tr = int(len(idx) * 0.8)
    tr_idx, va_idx = idx[:n_tr], idx[n_tr:]

    results = {"n_trajs": n_trajs}

    for name, model in [("jepa", jepa_model), ("rec", rec_model)]:
        print(f"  Encodage {name}…", end=" ", flush=True)
        Ztr, Ltr = _encode_trajs(model, tr_idx, dataset_dir, device)
        Zva, Lva = _encode_trajs(model, va_idx, dataset_dir, device)

        zm = Ztr.mean(0); zs = Ztr.std(0).clamp(min=1e-6)
        Ztr_n = (Ztr - zm) / zs
        Zva_n = (Zva - zm) / zs

        probe = nn.Linear(Ztr_n.shape[1], 2)
        opt   = torch.optim.Adam(probe.parameters(), lr=1e-2)
        ds_p  = torch.utils.data.TensorDataset(Ztr_n, Ltr)
        ld    = torch.utils.data.DataLoader(ds_p, batch_size=1024, shuffle=True)
        for _ in range(200):
            probe.train()
            for zb, lb in ld:
                loss = nn.functional.mse_loss(probe(zb), lb)
                opt.zero_grad(); loss.backward(); opt.step()

        probe.eval()
        with torch.no_grad():
            pred = probe(Zva_n)
        r2 = r2_score(pred, Lva)
        results[f"{name}_r2_theta"] = r2[0]
        results[f"{name}_r2_omega"] = r2[1]
        print(f"R²(θ)={r2[0]:.3f}  R²(ω)={r2[1]:.3f}")

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(results, indent=2))
    print(f"Probe mis en cache : {CACHE}")
    return results


# ── Dreaming ──────────────────────────────────────────────────────────────────

def _decode_chunked_jepa(decoder, z_traj, chunk=16):
    """z_traj : (n_steps+1, D)"""
    frames = []
    for s in range(0, z_traj.shape[0], chunk):
        out = decoder(z_traj[s:s+chunk])            # (chunk, 3, H, W)
        frames.append(out.clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
    return (np.concatenate(frames, axis=0) * 255).astype(np.uint8)


def _decode_chunked_rec(model, z_traj, chunk=16):
    """z_traj : (1, n_steps+1, D)"""
    frames = []
    for s in range(0, z_traj.shape[1], chunk):
        out = model.decode(z_traj[:, s:s+chunk])    # (1, chunk, 3, H, W)
        frames.append(out[0].clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy())
    return (np.concatenate(frames, axis=0) * 255).astype(np.uint8)


@torch.no_grad()
def build_dreams(jepa_model, jepa_decoder, rec_model,
                 frames_tensor: torch.Tensor, n_steps: int, device):
    """
    frames_tensor : (1, T, 3, 64, 64)
    Retourne : real_np, jepa_np, rec_np  — chacun (N, H, W, 3) uint8

    Seules N_SEED frames sont données à l'encodeur. Le reste est rêvé
    entièrement par le predictor — z0 a shape (1, 1, D).
    """
    f    = frames_tensor.to(device)
    seed = f[:, :N_SEED]                               # (1, N_SEED, 3, H, W)

    z_jepa = jepa_model.encode(seed)                   # (1, N_SEED, D)
    z_rec  = rec_model.encode(seed)                    # (1, N_SEED, D)

    z0_jepa = z_jepa[:, -1:]                           # (1, 1, D)
    z0_rec  = z_rec[:, -1:]                            # (1, 1, D)

    ztraj_jepa = jepa_model.imagine(z0_jepa, n_steps)  # (1, n_steps+1, D)
    ztraj_rec  = rec_model.imagine(z0_rec,  n_steps)   # (1, n_steps+1, D)

    jepa_np = _decode_chunked_jepa(jepa_decoder, ztraj_jepa[0])
    rec_np  = _decode_chunked_rec(rec_model, ztraj_rec)

    real_np = (f[0].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)

    return real_np, jepa_np, rec_np


# ── Viewer interactif ─────────────────────────────────────────────────────────

class CompareViewer:

    def __init__(self, jepa_model, jepa_decoder, rec_model,
                 dataset, bench, probe, args, device):
        self.jepa    = jepa_model
        self.jepa_d  = jepa_decoder
        self.rec     = rec_model
        self.dataset = dataset
        self.bench   = bench
        self.probe   = probe
        self.args    = args
        self.device  = device
        self.tdp     = device_tdp_watts(device)

        self.idx     = args.traj_idx if args.traj_idx >= 0 else random.randint(0, len(dataset) - 1)
        self.t       = 0
        self.playing = True

        self._load()
        self._build()
        self._start()

    # ── Chargement ──────────────────────────────────────────────────────────

    def _load(self):
        frames, _ = self.dataset[self.idx]          # (T, 3, 64, 64)
        n_steps   = min(self.args.n_steps, frames.shape[0] - N_SEED - 1)
        print(f"Traj {self.idx}  —  dreaming {n_steps} steps…", end=" ", flush=True)
        self.real_np, self.jepa_np, self.rec_np = build_dreams(
            self.jepa, self.jepa_d, self.rec,
            frames.unsqueeze(0), n_steps, self.device)
        print("ok")
        self.real_start = N_SEED - 1
        self.T = min(len(self.real_np) - self.real_start,
                     len(self.jepa_np), len(self.rec_np))
        self.t = 0

    # ── Figure ──────────────────────────────────────────────────────────────

    def _build(self):
        self.fig = plt.figure(figsize=(16, 6), facecolor=DARK)

        outer = gridspec.GridSpec(
            2, 1, figure=self.fig,
            height_ratios=[10, 1.2],
            hspace=0.10,
            left=0.02, right=0.98, top=0.91, bottom=0.05,
        )
        gs = gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=outer[0],
            wspace=0.05, width_ratios=[1, 1, 1, 0.9],
        )

        self.ax_real = self.fig.add_subplot(gs[0, 0])
        self.ax_jepa = self.fig.add_subplot(gs[0, 1])
        self.ax_rec  = self.fig.add_subplot(gs[0, 2])
        self.ax_info = self.fig.add_subplot(gs[0, 3])

        for ax, title, col in [
            (self.ax_real, "Réel",                  C_REAL),
            (self.ax_jepa, "JEPA imaginé",           C_JEPA),
            (self.ax_rec,  "AE imaginé",             C_AE),
        ]:
            ax.set_facecolor(DARK); ax.axis("off")
            ax.set_title(title, color=col, fontsize=11, pad=5)
            for sp in ax.spines.values():
                sp.set_edgecolor(col); sp.set_linewidth(1.5); sp.set_visible(True)

        self.ax_info.set_facecolor(DARK2); self.ax_info.axis("off")

        # Contrôles
        ctrl = gridspec.GridSpecFromSubplotSpec(
            1, 5, subplot_spec=outer[1],
            wspace=0.25, width_ratios=[1, 1, 1, 0.2, 5],
        )
        ax_prev  = self.fig.add_subplot(ctrl[0, 0])
        ax_play  = self.fig.add_subplot(ctrl[0, 1])
        ax_next  = self.fig.add_subplot(ctrl[0, 2])
        ax_slide = self.fig.add_subplot(ctrl[0, 4])

        from matplotlib.widgets import Button, Slider
        self.btn_prev = Button(ax_prev, "<  Prev", color="#222", hovercolor="#444")
        self.btn_play = Button(ax_play, "Pause",   color="#222", hovercolor="#444")
        self.btn_next = Button(ax_next, "Next  >", color="#222", hovercolor="#444")
        self.slider   = Slider(ax_slide, "Frame", 0, max(self.T - 1, 1),
                               valinit=0, valstep=1, color=C_REAL)

        for btn in (self.btn_prev, self.btn_play, self.btn_next):
            btn.label.set_color("white"); btn.label.set_fontsize(9)
        self.slider.label.set_color("white")
        self.slider.valtext.set_color("white")

        self.btn_prev.on_clicked(self._prev)
        self.btn_play.on_clicked(self._toggle)
        self.btn_next.on_clicked(self._next)
        self.slider.on_changed(self._on_slide)

        blank = np.zeros((64, 64, 3), dtype=np.uint8)
        self.im_real = self.ax_real.imshow(blank, interpolation="nearest")
        self.im_jepa = self.ax_jepa.imshow(blank, interpolation="nearest")
        self.im_rec  = self.ax_rec.imshow(blank,  interpolation="nearest")

        self._update_title()
        self._render_stats()
        self._draw(0)

    # ── Panel stats ──────────────────────────────────────────────────────────

    def _render_stats(self):
        ax = self.ax_info
        ax.clear(); ax.axis("off"); ax.set_facecolor(DARK2)

        b  = self.bench
        p  = self.probe
        W  = self.tdp

        def mj(ms):
            return ms_to_mj(ms, W)

        real_t  = min(self.real_start + self.t, len(self.real_np) - 1)
        dream_t = min(self.t, len(self.jepa_np) - 1)

        lines = []

        # ── Titre ──────────────────────────────────────────────────────────
        lines += [
            ("INFÉRENCE / STEP", "",  "white",  True),
            ("",                 "",  "white",  False),
        ]

        # ── JEPA ───────────────────────────────────────────────────────────
        lines += [
            ("JEPA",              "",  C_JEPA, True),
            ("  encode",  f"{b['jepa_encode_ms']:.1f} ms  "
                          f"~{mj(b['jepa_encode_ms']):.1f} mJ",  C_DIM, False),
            ("  rollout", f"{b['jepa_rollout_ms']:.1f} ms  "
                          f"~{mj(b['jepa_rollout_ms']):.1f} mJ", C_DIM, False),
            ("  decode",  f"{b['jepa_decode_ms']:.1f} ms  "
                          f"~{mj(b['jepa_decode_ms']):.1f} mJ",  C_DIM, False),
            ("  total",   f"{b['jepa_total_ms']:.1f} ms  "
                          f"~{mj(b['jepa_total_ms']):.1f} mJ",   C_JEPA, False),
            ("",          "",  "white", False),
        ]

        # ── AE ─────────────────────────────────────────────────────────────
        lines += [
            ("AE (Rec)",          "",  C_AE, True),
            ("  encode",  f"{b['rec_encode_ms']:.1f} ms  "
                          f"~{mj(b['rec_encode_ms']):.1f} mJ",   C_DIM, False),
            ("  rollout", f"{b['rec_rollout_ms']:.1f} ms  "
                          f"~{mj(b['rec_rollout_ms']):.1f} mJ",  C_DIM, False),
            ("  decode",  f"{b['rec_decode_ms']:.1f} ms  "
                          f"~{mj(b['rec_decode_ms']):.1f} mJ",   C_DIM, False),
            ("  total",   f"{b['rec_total_ms']:.1f} ms  "
                          f"~{mj(b['rec_total_ms']):.1f} mJ",    C_AE, False),
            ("",          "",  "white", False),
        ]

        # ── Probe ──────────────────────────────────────────────────────────
        lines += [
            ("PROBE  z → (θ,ω)", "",   "white", True),
            ("",                 "",   "white", False),
        ]
        if p:
            rj_t = p.get("jepa_r2_theta", float("nan"))
            rj_w = p.get("jepa_r2_omega", float("nan"))
            rr_t = p.get("rec_r2_theta",  float("nan"))
            rr_w = p.get("rec_r2_omega",  float("nan"))
            lines += [
                ("JEPA",
                 f"R²(θ)={rj_t:.3f}  R²(ω)={rj_w:.3f}", C_JEPA, False),
                ("AE",
                 f"R²(θ)={rr_t:.3f}  R²(ω)={rr_w:.3f}", C_AE,   False),
                ("",     "",  "white", False),
                ("→ θ",
                 "JEPA" if rj_t > rr_t else "AE", "white", False),
                ("→ ω",
                 "JEPA" if rj_w > rr_w else "AE", "white", False),
                ("",     "",  "white", False),
            ]

        # ── Frame info ─────────────────────────────────────────────────────
        lines += [
            ("t réel",   str(real_t),  C_REAL,  False),
            ("t imaginé", str(dream_t), C_DIM,   False),
            ("",          "",           "white",  False),
            (f"~{W:.0f}W (estimé)", "", "#555", False),
        ]

        y = 0.98
        dy = 0.048
        for label, val, color, bold in lines:
            w = "bold" if bold else "normal"
            ax.text(0.03, y, label, transform=ax.transAxes,
                    color=color, fontsize=7.5, fontweight=w, va="top",
                    fontfamily="monospace")
            if val:
                ax.text(0.52, y, val, transform=ax.transAxes,
                        color="white", fontsize=7.5, va="top",
                        fontfamily="monospace")
            y -= dy

    # ── Affichage ────────────────────────────────────────────────────────────

    def _draw(self, t):
        ri = self.real_start + t
        self.im_real.set_data(self.real_np[min(ri, len(self.real_np) - 1)])
        self.im_jepa.set_data(self.jepa_np[min(t,  len(self.jepa_np) - 1)])
        self.im_rec.set_data(self.rec_np[min(t,   len(self.rec_np) - 1)])
        self.slider.eventson = False
        self.slider.set_val(t)
        self.slider.eventson = True

    def _update_title(self):
        n = len(self.dataset)
        self.fig.suptitle(
            f"JEPA vs AE — trajectoire {self.idx + 1}/{n}  "
            f"(seed {N_SEED} frames)",
            color="white", fontsize=11, y=0.97,
        )

    # ── Animation ────────────────────────────────────────────────────────────

    def _animate(self, _):
        if self.playing:
            self.t = (self.t + 1) % self.T
            self._draw(self.t)
            if self.t % 5 == 0:
                self._render_stats()
        return [self.im_real, self.im_jepa, self.im_rec]

    def _start(self):
        interval = max(50, int(1000 / self.args.fps))
        self.anim = animation.FuncAnimation(
            self.fig, self._animate,
            interval=interval, blit=True, cache_frame_data=False,
        )
        plt.show()

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _toggle(self, _):
        self.playing = not self.playing
        self.btn_play.label.set_text("Pause" if self.playing else "Play")
        self.fig.canvas.draw_idle()

    def _on_slide(self, val):
        self.t = int(val)
        self._draw(self.t)
        self._render_stats()

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
        self._render_stats()
        self._draw(0)
        self.fig.canvas.draw_idle()


# ── Mode GIF ──────────────────────────────────────────────────────────────────

def save_gif(real_np, jepa_np, rec_np, real_start, path, fps):
    T = min(len(real_np) - real_start, len(jepa_np), len(rec_np))

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.5), facecolor=DARK)
    fig.patch.set_facecolor(DARK)
    for ax, title, col in zip(axes,
                               ["Réel", "JEPA imaginé", "AE imaginé"],
                               [C_REAL, C_JEPA, C_AE]):
        ax.set_facecolor(DARK); ax.axis("off")
        ax.set_title(title, color=col, fontsize=12)

    blank = np.zeros((64, 64, 3), dtype=np.uint8)
    im_r = axes[0].imshow(blank, interpolation="nearest")
    im_j = axes[1].imshow(blank, interpolation="nearest")
    im_a = axes[2].imshow(blank, interpolation="nearest")
    txt  = fig.text(0.5, 0.02, "t=0", ha="center", color="#999", fontsize=9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    def update(t):
        ri = min(real_start + t, len(real_np) - 1)
        im_r.set_data(real_np[ri])
        im_j.set_data(jepa_np[min(t, len(jepa_np) - 1)])
        im_a.set_data(rec_np[min(t, len(rec_np) - 1)])
        txt.set_text(f"t={t}  ({t * 0.05:.2f} s)")
        return [im_r, im_j, im_a, txt]

    anim = animation.FuncAnimation(fig, update, frames=T,
                                   interval=int(1000 / fps), blit=True)
    anim.save(path, fps=fps, writer="pillow")
    plt.close(fig)
    print(f"GIF sauvegardé : {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-ckpt",   default="checkpoints/jepa/lewm_best.pt")
    parser.add_argument("--decoder",     default="checkpoints/jepa/decoder_best.pt")
    parser.add_argument("--rec-ckpt",    default="checkpoints/rec/lewm_rec_best.pt")
    parser.add_argument("--dataset-dir", default="dataset/pendulum")
    parser.add_argument("--n-steps",     type=int,  default=100)
    parser.add_argument("--traj-idx",    type=int,  default=-1)
    parser.add_argument("--fps",         type=int,  default=12)
    parser.add_argument("--probe-trajs", type=int,  default=40,
                        help="Nb de trajectoires pour le probe rapide (0 = désactivé)")
    parser.add_argument("--no-cache",    action="store_true",
                        help="Recalcule le probe même si le cache existe")
    parser.add_argument("--gif",         action="store_true")
    parser.add_argument("--vis-dir",     default="visuals")
    args = parser.parse_args()

    device = get_device()
    print(f"Device : {device}")
    Path(args.vis_dir).mkdir(parents=True, exist_ok=True)
    if args.no_cache and CACHE.exists():
        CACHE.unlink()

    # ── Chargement des modèles ────────────────────────────────────────────
    print("Chargement des modèles…")
    jepa_model   = load_jepa(args.jepa_ckpt, device)
    jepa_decoder = load_jepa_decoder(args.decoder, jepa_model.embed_dim, device)
    rec_model    = load_rec(args.rec_ckpt, device)
    dataset      = PendulumSeqDataset(args.dataset_dir)
    print(f"  JEPA     : epoch={torch.load(args.jepa_ckpt, map_location='cpu', weights_only=False).get('epoch','?')}")
    print(f"  Decoder  : epoch={torch.load(args.decoder,   map_location='cpu', weights_only=False).get('epoch','?')}")
    print(f"  AE (Rec) : epoch={torch.load(args.rec_ckpt,  map_location='cpu', weights_only=False).get('epoch','?')}")

    # ── Benchmark d'inférence ─────────────────────────────────────────────
    print("\nBenchmark d'inférence…")
    frames0, _ = dataset[0]
    seed_frames = frames0[:N_SEED].unsqueeze(0).to(device)
    bench = benchmark_inference(
        jepa_model, jepa_decoder, rec_model,
        seed_frames, n_steps=args.n_steps, device=device,
    )

    tdp = device_tdp_watts(device)
    print(f"\n  TDP estimé : {tdp:.0f} W")
    print(f"  {'':8s}  {'JEPA':>12s}  {'AE':>12s}")
    for phase in ("encode", "rollout", "decode", "total"):
        j = bench[f"jepa_{phase}_ms"]
        r = bench[f"rec_{phase}_ms"]
        print(f"  {phase:8s}  {j:9.2f} ms  {r:9.2f} ms")

    # ── Probe rapide ──────────────────────────────────────────────────────
    probe = None
    if args.probe_trajs > 0:
        print(f"\nProbe linéaire ({args.probe_trajs} trajectoires)…")
        probe = quick_probe(jepa_model, rec_model, args.dataset_dir,
                            args.probe_trajs, device)

    # ── Lancement ─────────────────────────────────────────────────────────
    if args.gif:
        idx     = args.traj_idx if args.traj_idx >= 0 else 0
        frames, _ = dataset[idx]
        n_steps   = min(args.n_steps, frames.shape[0] - N_SEED - 1)
        real_np, jepa_np, rec_np = build_dreams(
            jepa_model, jepa_decoder, rec_model,
            frames.unsqueeze(0), n_steps, device)
        gif_path = f"{args.vis_dir}/compare_{idx:04d}.gif"
        save_gif(real_np, jepa_np, rec_np, N_SEED - 1, gif_path, args.fps)
    else:
        CompareViewer(jepa_model, jepa_decoder, rec_model,
                      dataset, bench, probe, args, device)


if __name__ == "__main__":
    main()
