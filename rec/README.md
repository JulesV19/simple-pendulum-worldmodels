# rec/

Pipeline d'entraînement et d'inférence pour **LeWorldModelRec** (autoencoder + SIGReg).

---

## Architecture

```
(frame_t, diff_t)  [6 canaux]
        │
   CNN encoder  →  z ∈ R^128
        │
        ├──► Decoder  →  frame_hat        (reconstruction)
        │         loss: MSE + VGG + FFT
        │
        └──► MLP predictor  →  ẑ_{t+k}   k = 1 … rollout_k
                    │
               Decoder  →  frame_pred_{t+k}    (prédiction)
                         loss: MSE + VGG + FFT

loss = rec_coef·MSE_rec + pred_coef·MSE_pred
     + perceptual_coef·(VGG_rec + VGG_pred)
     + freq_coef·(FFT_rec + FFT_pred)
     + λ·SIGReg
```

**Différences vs JEPA :**
- Pas de target encoder EMA → pas de risque de collapse silencieux
- Supervision pixel directe → représentation plus interprétable, meilleur R²(θ)
- Le décodeur est partagé entre reconstruction et prédiction → cohérence de l'espace latent
- Le diff de frames encode ω par effet de bord, sans la pression explicite du rollout latent

**Anti-flou :** MSE seul produit des frames floues (solution = moyenne des modes). `PerceptualLoss` (VGG16) et `FrequencyLoss` (FFT 2D) corrigent cela. Voir [`models/README.md`](../models/README.md) pour les détails.

---

## Entraînement

```bash
# Lancement de base
python3 rec/train.py

# Avec hyperparamètres
python3 rec/train.py --epochs 50 --batch-size 16

# Reprendre depuis un checkpoint
python3 rec/train.py --checkpoint checkpoints/rec/lewm_rec_best.pt
```

**Hyperparamètres clés :**

| Paramètre | Défaut | Rôle |
|---|---|---|
| `embed_dim` | 128 | dimension de l'espace latent |
| `rollout_k` | 10 | horizon de prédiction (~0.5 s, aligné JEPA) |
| `rec_coef` | 1.0 | poids reconstruction MSE |
| `pred_coef` | 1.0 | poids prédiction MSE |
| `perceptual_coef` | 0.1 | poids VGG16 — 0 = désactivé |
| `freq_coef` | 0.05 | poids FFT — 0 = désactivé |
| `lam` | 0.1 | poids SIGReg |
| `batch_size` | 16 | réduit vs JEPA (le décodeur est dans le graph) |

Le `batch_size=16` (vs 32 pour JEPA) est une contrainte mémoire : le décodeur fait tourner des ConvTranspose pour chaque step du rollout, ce qui double l'empreinte GPU par rapport au JEPA.

Checkpoints sauvés dans `checkpoints/rec/` : `lewm_rec_best.pt`, `lewm_rec_last.pt`.

---

## Inférence

### Imagination (`imagine.py`)

Affiche trois colonnes en parallèle : frame réelle, reconstruction `decode(encode(frame))`, et frame imaginée `decode(predictor^k(z_0))`.

```bash
python3 rec/imagine.py
python3 rec/imagine.py --n-steps 200 --gif
python3 rec/imagine.py --traj-idx 3
```

La qualité visuelle de la reconstruction est un bon proxy pour savoir si l'encodeur a convergé — si la reconstruction est floue, la pred loss est aussi affectée.

---

## Fichiers

| Fichier | Rôle |
|---|---|
| `train.py` | boucle d'entraînement AE |
| `imagine.py` | viewer réel / reconstruction / imaginé |
| `notebooks/` | version Colab |
