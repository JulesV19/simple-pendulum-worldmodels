# models/

Composants partagés entre les deux architectures (JEPA et AE).

---

## Encoder — `encoder.py`

CNN déterministe : `(B, 6, 64, 64) → R^embed_dim`.

L'entrée est toujours **6 canaux** : `concat(frame_t, frame_t − frame_{t-1})`. Le diff donne au réseau un accès direct à la vitesse angulaire ω sans supervision explicite.

```
(B, 6, 64, 64)
  Conv2d(6→32,  stride=2)  →  (32, 32, 32)
  Conv2d(32→64, stride=2)  →  (64, 16, 16)
  Conv2d(64→128,stride=2)  →  (128, 8,  8)
  Conv2d(128→256,stride=2) →  (256, 4,  4)
  Linear(256×4×4 → embed_dim)
```

**`ContextEncoder`** — encodeur principal, gradient actif.  
**`TargetEncoder`** — copie EMA du `ContextEncoder`, aucun gradient. Utilisé uniquement par JEPA pour fournir des cibles stables au predictor sans collapse.

```python
# Mise à jour EMA (après chaque optimizer.step())
model.target_encoder.update(model.encoder)

# Momentum par défaut : 0.996
# → les poids du target bougent ~0.4% par step
```

---

## Decoder — `decoder.py`

Miroir symétrique du CNN encoder : `R^embed_dim → (B, 3, 64, 64)`.

**L'entrée z est normalisée (L2) avant le forward.** Cela rend le décodeur invariant à la magnitude de z — indispensable en imagination où le predictor peut faire dériver la norme de z au fil des steps.

```
z ∈ R^embed_dim
  L2-normalize
  Linear(embed_dim → 256×4×4)  →  reshape (256, 4, 4)
  ConvTranspose(256→128, stride=2)  →  (128, 8,  8)
  ConvTranspose(128→64,  stride=2)  →  (64,  16, 16)
  ConvTranspose(64→32,   stride=2)  →  (32,  32, 32)
  ConvTranspose(32→3,    stride=2)  →  (3,   64, 64)
  Sigmoid → [0, 1]
```

Accepte `(B, D)` et `(B, T, D)` — retourne respectivement `(B, 3, 64, 64)` et `(B, T, 3, 64, 64)`.

---

## SIGReg — `sigreg.py`

Régulariseur qui pousse la distribution des embeddings vers `N(0, I)`.

**Principe (théorème de Cramér-Wold) :** une distribution est gaussienne isotrope si et seulement si toute projection 1D sur un vecteur aléatoire `u ~ Uniform(S^{d-1})` suit une loi `N(0, 1)`. SIGReg applique le test de normalité d'Epps-Pulley à `n_proj=512` projections aléatoires.

**Statistique d'Epps-Pulley :**
```
T(h) = (1/N²) Σᵢ Σⱼ exp(−(hᵢ−hⱼ)²/2)
      − √2 · (1/N) Σᵢ exp(−hᵢ²/4)
      + 1/√3
```
`T → 0` si `h ~ N(0,1)`, `T > 0` sinon.

Deux termes supplémentaires contre le collapse dimensionnel (invisible à Epps-Pulley si la variance est concentrée dans quelques dims) :

| Terme | Rôle |
|---|---|
| `mean_penalty = E[z_d]².mean()` | pousse la moyenne vers 0 |
| `var_penalty = (std(z_d) − 1)².mean()` | pousse chaque dimension à std=1 |

**Complexité :** `O(N² × n_proj)` — N est sous-échantillonné à `max_n=256`.

**Référence :** Maes et al., *LeWorldModel*, arXiv:2603.19312 (2026).

---

## Losses — `losses.py`

Deux pertes auxiliaires pour lutter contre le flou MSE, utilisées uniquement par l'AE.

### `PerceptualLoss` (VGG16)

Compare les feature maps à trois profondeurs du VGG16 pré-entraîné (gelé) :

| Couche | Résolution | Capture |
|---|---|---|
| `relu1_2` | 64×64 | bords, couleurs |
| `relu2_2` | 32×32 | textures |
| `relu3_3` | 16×16 | structures |

MSE minimise `E[(p−t)²]` dont la solution optimale est la moyenne des modes → flou. Dans l'espace VGG, les activations sont éparses et non-linéaires, donc la "moyenne" n'est plus une solution facile.

Les frames sont normalisées ImageNet en interne. Un sous-échantillonnage à `max_samples=64` évite l'OOM lors des rollouts longs.

### `FrequencyLoss` (FFT 2D)

Pénalise les différences de spectre (amplitude + phase) entre pred et target. Complémentaire à MSE : MSE est aveugle aux hautes fréquences dont la contribution L2 est faible mais l'impact perceptuel est fort.

Option `high_freq_boost=True` : multiplie le spectre par une rampe radiale proportionnelle à la fréquence spatiale → amplifie encore la netteté. Pas de dépendances externes (`torch.fft`).

---

## Architectures complètes

| | **`jepa/model.py`** | **`rec/model.py`** |
|---|---|---|
| Classe | `LeWorldModel` | `LeWorldModelRec` |
| Encoder | `ContextEncoder` + `TargetEncoder` (EMA) | `ContextEncoder` seul |
| Decoder | aucun (entraîné séparément) | `Decoder` intégré |
| Predictor | `TransitionPredictor` (MLP résiduel) | `TransitionPredictor` (identique) |
| Supervision | cosine + MSE dans l'espace latent | MSE + VGG + FFT dans le pixel space |
| SIGReg | oui | oui |

Voir les READMEs de [`../jepa/`](../jepa/README.md) et [`../rec/`](../rec/README.md) pour les détails d'entraînement.
