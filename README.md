# JEPA vs Autoencoder World Model

Comparaison empirique de deux approches de world model sur pendule simple :
**JEPA** (prédiction dans l'espace latent) contre une **architecture autoencoder classique** (reconstruction pixel).

| | **JEPA** | **AE (Rec)** |
|---|---|---|
| Supervision | espace latent (cosine + MSE) | pixel space (MSE + VGG + FFT) |
| Target encoder | EMA | aucun |
| Décodeur | entraîné séparément | intégré, entraîné conjointement |
| Probe R²(θ) | 0.966 | 0.976 |
| Probe R²(ω) | 0.918 | 0.905 |

> **Références :**
> Maes et al., *"Le World Model"*, arXiv:2603.19312 (2026) · LeCun, *"A Path Towards Autonomous Machine Intelligence"* (2022) · Assran et al., *I-JEPA* (CVPR 2023)

---

## Architectures

### JEPA — `models/jepa/model.py`

```
(frame_t, diff_t)  [6ch]
       ↓
  CNN encoder online  →  z_ctx ∈ R^128   [gradient]
  CNN encoder target  →  z_tgt ∈ R^128   [EMA, no grad]
       ↓
  MLP predictor  →  ẑ_{t+k}   k = 1…rollout_k

loss = (1/K) Σ [ cosine(ẑ_{t+k}, z*_{t+k}) + α·MSE ] + λ·SIGReg
```

Le MLP ne voit que `z_t` — sans `ω` dans `z`, prédire `z_{t+1}` est impossible.

### AE — `models/rec/model.py`

```
(frame_t, diff_t)  [6ch]
       ↓
  CNN encoder  →  z ∈ R^128
       ↓                    ↓
  Decoder  →  frame_hat    MLP predictor  →  ẑ_{t+k}
                                               ↓
                                           Decoder  →  frame_pred_{t+k}

loss = MSE(frame_hat, frame) + perceptual(VGG) + freq(FFT) + λ·SIGReg
```

### Composants partagés — `models/`

| Fichier | Rôle |
|---|---|
| `encoder.py` | `ContextEncoder` (CNN 4 couches) + `TargetEncoder` (EMA) |
| `decoder.py` | `Decoder` z → frame (ConvTranspose ×4) |
| `sigreg.py` | SIGReg — Epps-Pulley test, force z ~ N(0, I) |
| `losses.py` | `PerceptualLoss` (VGG16) + `FrequencyLoss` (FFT 2D) |

---

## Dataset

Pendule simple, 64×64 px, `states = [θ, ω]`.

```bash
python3 data/generate.py --n_trajectories 2000 --n_frames 500
# → dataset/pendulum/traj_XXXX.npz  :  frames (T,H,W,3)  +  states (T,2)
```

```bash
python3 tools/browse.py        # navigateur interactif + portrait de phase
python3 tools/visualize.py     # grid de trajectoires
```

---

## JEPA

### Entraînement

```bash
# Local
python3 jepa/train.py
python3 jepa/train.py --lam 1.0 --rollout-k 20 --epochs 50
python3 jepa/train.py --checkpoint checkpoints/jepa/lewm_best.pt   # resume

# Décodeur séparé (encodeur gelé)
python3 jepa/train_decoder.py --checkpoint checkpoints/jepa/lewm_best.pt
```

Colab : `jepa/notebooks/train_colab.ipynb` · `jepa/notebooks/train_decoder_colab.ipynb`

**Hyperparamètres clés :**

| Paramètre | Valeur | Note |
|---|---|---|
| `embed_dim` | 128 | dimension latente |
| `rollout_k` | 20 | ~1 s ≈ demi-période du pendule |
| `lam` | 1.0 | poids SIGReg |
| `mse_coef` | 0.1 | poids MSE dans pred_loss |
| `ema_momentum` | 0.996 | EMA target encoder |

### Inférence

```bash
python3 jepa/imagine.py
python3 jepa/imagine.py --n-steps 200 --gif
python3 jepa/eval.py --checkpoint checkpoints/jepa/lewm_best.pt
```

---

## AE (LeWorldModelRec)

### Entraînement

```bash
# Local
python3 rec/train.py
python3 rec/train.py --epochs 50 --batch-size 16
python3 rec/train.py --checkpoint checkpoints/rec/lewm_rec_best.pt   # resume
```

Colab : `rec/notebooks/train_colab.ipynb`

**Hyperparamètres clés :**

| Paramètre | Valeur | Note |
|---|---|---|
| `embed_dim` | 128 | dimension latente |
| `rollout_k` | 5 | horizon de prédiction |
| `rec_coef` | 1.0 | poids reconstruction MSE |
| `perceptual_coef` | 0.1 | poids VGG16 — anti-flou |
| `freq_coef` | 0.05 | poids FFT — anti-flou |
| `batch_size` | 16 | réduit vs JEPA (décodeur dans le graph) |

### Inférence

```bash
python3 rec/imagine.py                  # réel / reconstruction / imaginé
python3 rec/imagine.py --n-steps 200 --gif
```

---

## Évaluation comparative

### Probe linéaire z → (θ, ω)

```bash
# Comparaison directe sur les deux meilleurs checkpoints
python3 eval/probe.py --compare

# Probe sur un modèle seul
python3 eval/probe.py --model jepa --checkpoint checkpoints/jepa/lewm_best.pt
python3 eval/probe.py --model rec  --checkpoint checkpoints/rec/lewm_rec_best.pt

# Sample efficiency (probe entraîné sur 10% des données)
python3 eval/probe.py --compare --label-frac 0.1

# Sweep sur un dossier de checkpoints → courbe R² vs epoch
python3 eval/probe.py --model jepa --checkpoint-dir checkpoints/jepa/ --plot
```

Le R²(ω) est la métrique clé : il mesure si la dynamique est encodée dans z,
indépendamment de la qualité visuelle.

### Visualisation côte-à-côte

```bash
python3 eval/compare.py                 # viewer JEPA vs AE + stats
python3 eval/compare.py --probe-trajs 0 # sans recalcul du probe
python3 eval/compare.py --gif
```

### Scatter plots

```bash
python3 eval/scatter.py --checkpoint checkpoints/jepa/lewm_best.pt
```

---

## Structure

```
WorldModel/
├── data/
│   ├── dataset.py          PendulumSeqDataset, dataloaders
│   └── generate.py         génération dataset
│
├── models/
│   ├── encoder.py          CNN 4 couches (partagé)
│   ├── decoder.py          ConvTranspose 4 couches (partagé)
│   ├── sigreg.py           SIGReg regularizer
│   ├── losses.py           PerceptualLoss (VGG16), FrequencyLoss (FFT)
│   ├── jepa/model.py       LeWorldModel
│   └── rec/model.py        LeWorldModelRec
│
├── jepa/
│   ├── train.py
│   ├── train_decoder.py
│   ├── imagine.py
│   ├── eval.py
│   └── notebooks/
│
├── rec/
│   ├── train.py
│   ├── imagine.py
│   └── notebooks/
│
├── eval/
│   ├── probe.py            comparaison R²(θ,ω)
│   ├── scatter.py          scatter linéaire vs MLP
│   └── compare.py          viewer côte-à-côte + benchmark
│
├── tools/
│   ├── browse.py           navigateur dataset
│   └── visualize.py        viewer trajectoires
│
└── checkpoints/
    ├── jepa/               lewm_best.pt, decoder_best.pt
    └── rec/                lewm_rec_best.pt
```

---

## Résultats probe (val set, 200 epochs)

```
                    JEPA      AE
R²(θ)             0.966     0.976    AE +1pt   — AE encode mieux la position
R²(ω)             0.918     0.905    JEPA +1pt — JEPA encode mieux la vitesse
R²(mean)          0.942     0.940    quasi-identique
```

R²(ω) est la signature de la qualité dynamique : JEPA encode ω sans supervision
pixel grâce au rollout_k=20 qui force le predictor à résoudre la dynamique.
L'AE encode ω par effet de bord du diff de frames, mais avec moins de pression.

> **Limite de cette comparaison :** rollout_k=20 (JEPA) vs 5 (AE), batch_size=32 vs 16.
> Pour une comparaison contrôlée, aligner ces hyperparamètres et tracer R² vs gradient steps.
