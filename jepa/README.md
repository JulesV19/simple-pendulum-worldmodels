# jepa/

Pipeline d'entraînement et d'inférence pour **LeWorldModel** (JEPA + SIGReg).

---

## Architecture

```
(frame_t, diff_t)  [6 canaux]
        │
        ├──► CNN encoder online  →  z_ctx ∈ R^128   [gradient actif]
        └──► CNN encoder target  →  z_tgt ∈ R^128   [EMA, no grad]
                                          │
                    MLP predictor ◄────── z_ctx_t
                         │
                         ▼
                    ẑ_{t+k}   k = 1 … rollout_k

loss = (1/K) Σₖ [ cosine(ẑ_{t+k}, z*_{t+k}) + α·MSE + β·norm ] + λ·SIGReg
```

**Pourquoi le MLP force ω dans z :** le predictor ne voit que `z_t` — sans la vitesse angulaire encodée dans `z_t`, prédire `z_{t+1}` est impossible (même position, deux directions possibles). Un Transformer aurait pu calculer `z_t − z_{t-1} ≈ ω` lui-même, court-circuitant l'encodeur.

**Pourquoi cosine + MSE :** la cosine distance évite le raccourci "moyenne des modes ±ω". Le terme MSE contraint la magnitude de `z_roll` pour que le décodeur (entraîné séparément) reste valide.

---

## Entraînement

```bash
# Lancement de base
python3 jepa/train.py

# Avec hyperparamètres
python3 jepa/train.py --lam 0.5 --rollout-k 10 --epochs 50

# Reprendre depuis un checkpoint
python3 jepa/train.py --checkpoint checkpoints/jepa/lewm_best.pt
```

**Hyperparamètres clés :**

| Paramètre | Défaut | Rôle |
|---|---|---|
| `embed_dim` | 128 | dimension de l'espace latent |
| `rollout_k` | 10 | horizon de prédiction (~0.5 s) |
| `lam` | 0.5 | poids SIGReg |
| `mse_coef` | 0.1 | poids MSE dans la pred loss |
| `norm_coef` | 1.0 | conservation de norme pendant le rollout |
| `ema_momentum` | 0.996 | lenteur de mise à jour du target encoder |
| `batch_size` | 32 | |

Le target encoder doit être mis à jour **après chaque `optimizer.step()`** via `model.update_target()`. Oublier cet appel = target encoder figé = collapse.

Checkpoints sauvés dans `checkpoints/jepa/` : `lewm_best.pt` (meilleure val loss), `lewm_last.pt`.

---

## Décodeur séparé

L'encodeur JEPA ne voit jamais de supervision pixel pendant l'entraînement principal. Pour visualiser les frames imaginées, un décodeur est entraîné **séparément**, encodeur gelé :

```bash
python3 jepa/train_decoder.py --checkpoint checkpoints/jepa/lewm_best.pt
```

Le décodeur entraîné est sauvé dans `checkpoints/jepa/decoder_best.pt`.

**Pourquoi séparé :** si le décodeur était entraîné conjointement, la reconstruction pixel deviendrait un signal de supervision supplémentaire pour l'encodeur — on perdrait la pureté JEPA (pas de supervision pixel).

---

## Inférence

### Imagination (`imagine.py`)

Encode les 2 premières frames réelles (pour avoir `diff ≠ 0`, donc ω dans z), puis roule le predictor sans jamais revoir les frames suivantes.

```bash
python3 jepa/imagine.py
python3 jepa/imagine.py --n-steps 200 --gif
python3 jepa/imagine.py --traj-idx 5
```

Viewer interactif : Espace = play/pause, slider = scrubbing, boutons Prev/Next = changer de trajectoire.

### Évaluation probe (`eval.py`)

Régression linéaire `z → (θ, ω)` pour mesurer si l'espace latent encode les états physiques sans supervision.

```bash
python3 jepa/eval.py --checkpoint checkpoints/jepa/lewm_best.pt
```

Pour la comparaison directe JEPA vs AE, voir [`eval/probe.py`](../eval/probe.py).

---

## Fichiers

| Fichier | Rôle |
|---|---|
| `train.py` | boucle d'entraînement JEPA |
| `train_decoder.py` | entraînement du décodeur (encodeur gelé) |
| `imagine.py` | viewer réel vs imaginé |
| `eval.py` | probe linéaire z → (θ, ω) |
| `notebooks/` | versions Colab |
