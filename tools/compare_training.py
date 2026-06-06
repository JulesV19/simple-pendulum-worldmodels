"""
Visualisation comparative des stats d'entraînement JEPA vs REC.
Usage: python compare_training.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

ROOT = Path(__file__).parent
JEPA = json.loads((ROOT / "training_stats_jepa.json").read_text())
REC  = json.loads((ROOT / "training_stats_rec.json").read_text())

COLORS = {"jepa": "#4C72B0", "rec": "#DD8452"}
EPOCHS = list(range(1, 51))

# ── helpers ──────────────────────────────────────────────────────────────────

def plot_loss_pair(ax, jepa_seq, rec_seq, title, ylabel="loss", logy=False):
    ax.plot(EPOCHS, jepa_seq, color=COLORS["jepa"], label="JEPA")
    ax2 = ax.twinx()
    ax2.plot(EPOCHS, rec_seq, color=COLORS["rec"], label="REC", linestyle="--")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel + " (JEPA)", color=COLORS["jepa"])
    ax2.set_ylabel(ylabel + " (REC)", color=COLORS["rec"])
    ax.tick_params(axis="y", labelcolor=COLORS["jepa"])
    ax2.tick_params(axis="y", labelcolor=COLORS["rec"])
    if logy:
        ax.set_yscale("log")
        ax2.set_yscale("log")
    lines = [plt.Line2D([0], [0], color=COLORS["jepa"], label="JEPA"),
             plt.Line2D([0], [0], color=COLORS["rec"], linestyle="--", label="REC")]
    ax.legend(handles=lines, fontsize=8)


def plot_single(ax, seq, label, color, title, ylabel="loss", linestyle="-"):
    ax.plot(EPOCHS, seq, color=color, label=label, linestyle=linestyle)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)


def power_hist(ax, readings, label, color):
    ax.hist(readings, bins=30, color=color, alpha=0.7, label=label, edgecolor="white", linewidth=0.3)
    ax.axvline(np.mean(readings), color=color, linestyle="--", linewidth=1.5, label=f"mean={np.mean(readings):.1f}W")
    ax.set_xlabel("Power (W)")
    ax.set_ylabel("Count")
    ax.legend(fontsize=8)

# ── figure layout ─────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(18, 14))
fig.suptitle("JEPA vs REC — Comparaison des entraînements", fontsize=14, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.42, wspace=0.38)

# Row 0 — convergence
ax00 = fig.add_subplot(gs[0, :2])  # JEPA train+val
ax01 = fig.add_subplot(gs[0, 2:])  # REC train+val

# Row 1 — pred loss, sigreg, power histos
ax10 = fig.add_subplot(gs[1, :2])  # pred loss comparison
ax11 = fig.add_subplot(gs[1, 2:])  # sigreg comparison

# Row 2 — power + summary
ax20 = fig.add_subplot(gs[2, 0])   # JEPA power histo
ax21 = fig.add_subplot(gs[2, 1])   # REC power histo
ax22 = fig.add_subplot(gs[2, 2:])  # summary table

# ── (0,0) JEPA convergence ───────────────────────────────────────────────────
j_cv = JEPA["convergence"]
ax00.plot(EPOCHS, j_cv["train_loss"], color=COLORS["jepa"], label="train")
ax00.plot(EPOCHS, j_cv["val_loss"],   color=COLORS["jepa"], linestyle="--", alpha=0.7, label="val")
best_e = j_cv["best_epoch"]
ax00.axvline(best_e, color="gray", linestyle=":", linewidth=1, label=f"best epoch {best_e}")
ax00.set_title("JEPA — Convergence (train/val)", fontsize=10, fontweight="bold")
ax00.set_xlabel("Epoch"); ax00.set_ylabel("Loss")
ax00.legend(fontsize=8)

# ── (0,1) REC convergence ────────────────────────────────────────────────────
r_cv = REC["convergence"]
ax01.plot(EPOCHS, r_cv["train_loss"], color=COLORS["rec"], label="train")
ax01.plot(EPOCHS, r_cv["val_loss"],   color=COLORS["rec"], linestyle="--", alpha=0.7, label="val")
best_e_r = r_cv["best_epoch"]
ax01.axvline(best_e_r, color="gray", linestyle=":", linewidth=1, label=f"best epoch {best_e_r}")
ax01.set_title("REC — Convergence (train/val)", fontsize=10, fontweight="bold")
ax01.set_xlabel("Epoch"); ax01.set_ylabel("Loss")
ax01.legend(fontsize=8)

# ── (1,0) Pred loss ──────────────────────────────────────────────────────────
plot_loss_pair(ax10, j_cv["pred_loss"], r_cv["pred_loss"],
               "Pred Loss (axes séparés — échelles différentes)")

# ── (1,1) SigReg ─────────────────────────────────────────────────────────────
plot_loss_pair(ax11, j_cv["sigreg"], r_cv["sigreg"],
               "SigReg (axes séparés)")

# ── (2,0-1) Power histograms ─────────────────────────────────────────────────
power_hist(ax20, JEPA["energy"]["power_readings_W"], "JEPA", COLORS["jepa"])
ax20.set_title("JEPA — Distribution puissance GPU", fontsize=10, fontweight="bold")

power_hist(ax21, REC["energy"]["power_readings_W"], "REC", COLORS["rec"])
ax21.set_title("REC — Distribution puissance GPU", fontsize=10, fontweight="bold")

# ── (2,2) Summary table ───────────────────────────────────────────────────────
ax22.axis("off")
col_labels = ["Métrique", "JEPA", "REC"]
rows = [
    ["Params (M)",         f"{JEPA['memory']['model_params_M']:.3f}",     f"{REC['memory']['model_params_M']:.3f}"],
    ["Batch size",         str(JEPA["hyperparams"]["batch_size"]),         str(REC["hyperparams"]["batch_size"])],
    ["Grad steps",         str(JEPA["time"]["gradient_steps"]),            str(REC["time"]["gradient_steps"])],
    ["Durée totale (min)", f"{JEPA['time']['total_min']:.1f}",             f"{REC['time']['total_min']:.1f}"],
    ["Temps / epoch (s)",  f"{JEPA['time']['avg_per_epoch_s']:.1f}",       f"{REC['time']['avg_per_epoch_s']:.1f}"],
    ["Peak GPU (MB)",      f"{JEPA['memory']['peak_gpu_MB']:.0f}",         f"{REC['memory']['peak_gpu_MB']:.0f}"],
    ["Consomm. (Wh)",      f"{JEPA['energy']['total_Wh']:.2f}",           f"{REC['energy']['total_Wh']:.2f}"],
    ["Puissance moy. (W)", f"{JEPA['energy']['avg_power_W']:.1f}",        f"{REC['energy']['avg_power_W']:.1f}"],
    ["Best val loss",      f"{JEPA['convergence']['best_val_loss']:.6f}",  f"{REC['convergence']['best_val_loss']:.6f}"],
    ["Best epoch",         str(JEPA["convergence"]["best_epoch"]),         str(REC["convergence"]["best_epoch"])],
]

# JEPA-specific extras
if "probe_linear" in JEPA:
    rows.append(["R² θ (linear probe)", f"{JEPA['probe_linear']['r2_theta']:.4f}", "—"])
    rows.append(["R² ω (linear probe)", f"{JEPA['probe_linear']['r2_omega']:.4f}", "—"])
if "rollout_latent" in JEPA:
    rows.append(["Cosine sim step 10", f"{JEPA['rollout_latent']['cosine_sim_step10']:.4f}", "—"])
    rows.append(["Cosine sim step 20", f"{JEPA['rollout_latent']['cosine_sim_step20']:.4f}", "—"])

table = ax22.table(
    cellText=rows, colLabels=col_labels,
    loc="center", cellLoc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.35)

for (r, c), cell in table.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2d2d2d")
        cell.set_text_props(color="white", fontweight="bold")
    elif c == 1:
        cell.set_facecolor("#dce9f5")
    elif c == 2:
        cell.set_facecolor("#fce8d5")
    if c == 0 and r > 0:
        cell.set_text_props(ha="left")

ax22.set_title("Résumé comparatif", fontsize=10, fontweight="bold", pad=8)

# ── save & show ───────────────────────────────────────────────────────────────
out = ROOT / "training_comparison.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Sauvegardé : {out}")
plt.show()
