# Team Roadmap

> This file is written to be fed into your AI coding agent.
> Copy the relevant section (shared orientation + your owner block) into a fresh conversation and the agent should have enough context to implement without asking questions.

## Quick orientation — decisions already locked, do NOT revisit

- **Backbone is fixed**: custom 3-conv CNN trained from scratch, sliced at `Dense(128, relu)`. **NOT ResNet18. NOT pretrained.** Defined in `privacy_ml/models.py`.
- **Embedding dim**: 128.
- **Input**: 150×150×1 grayscale images.
- **Dataset**: Kaggle `paultimothymooney/chest-xray-pneumonia`. Loader: `privacy_ml.data.load_kaggle_pool`. The Kaggle zip extracts to a nested `chest_xray/chest_xray/` layout — auto-resolved by `resolve_kaggle_base`.
- **Canonical splits**: `split_pool_indices(y, seed=42)` → `victim_members` (2000) + `victim_nonmembers` (1000) + `shadow_pool` (2856), stratified on class.
- **Split-model architecture** (presentation slide 3):
  ```
  image (150×150×1)
    → [BIE?]                ← image-layer PPT (İrem Damla)
    → encoder (3-conv CNN, frozen once trained)
    → 128-d embedding
    → [DP?]                 ← embedding-layer PPT (Onur)
    → [SMPC?]               ← embedding-layer PPT (Eray)
    → head (Dense-1, sigmoid)
    → confidence ∈ [0, 1]
  ```
- **PPT contract**: `privacy_ml/ppt/base.py :: PrivacyMechanism`. Implement `layer: Literal["image","embedding"]`, `fit(X) -> None`, `apply(X) -> ndarray`. Input/output shapes must match.
- **Authoritative documents**:
  - Design spec: `docs/superpowers/specs/2026-04-18-mia-design.md`
  - Protocol source: `privacy_ml/ppt/base.py`
  - Reference implementations (no-op stubs): `privacy_ml/ppt/stubs.py`

## Overall order of operations

```
Phase A: Contract setup  ← DONE (Egehan, commits b6001c1..58f13f7)
  ✅ Design spec + audit
  ✅ models.py + data.py + tests
  ✅ Kaggle-layout auto-probe
  ✅ ppt/base.py + ppt/stubs.py + cache.py + tests

Phase B: Parallel implementation  ← WE ARE HERE
  3. [Egehan]      Step 3–5: attacks, metrics, runner, CLI
  4. [Onur]        DP module         → privacy_ml/ppt/dp.py
  5. [Eray]        SMPC module       → privacy_ml/ppt/smpc.py
  6. [İrem Damla]  BIE module        → privacy_ml/ppt/bie.py
  7. [İlmay]       Reconstruction attack → reconstruction_attack.ipynb (already scaffolded, using privacy_ml)
  8. [All]         Lit-review paragraphs in conference_101719.tex

Phase C: Integration & experiments
  9.  [Egehan] Run 8 canonical configs on Colab → results/runs.jsonl
  10. [İlmay]  Run reconstruction evals in parallel
  11. [Egehan] Generate Results table + figures

Phase D: Paper
  12. [All] Methods + Results + Discussion sections
  13. [All] Update slide 3 wording ("custom CNN", not "pretrained backbone") for final presentation
  14. [All] Proofread + submit
```

## Environment setup (everyone)

```bash
# one time
git clone https://github.com/Erayisci/Privacy_Preserving_ML.git
cd Privacy_Preserving_ML
pip install -e '.[dev]'           # deps + pytest
pytest tests/ -v                  # ~54 tests without TF (cache, data, ppt, bie); test_models needs TensorFlow

# every work session
git pull --rebase origin main
# ... implement + test ...
pytest tests/ -v
git add privacy_ml/ppt/yourmodule.py tests/test_yourmodule.py
git commit -m "Implement <yourmodule>"
git push origin main              # rebase if rejected — we keep main linear
```

## Your task — find your owner block below

### Onur — Differential Privacy (DP)

**Create**: `privacy_ml/ppt/dp.py`
**Layer**: `"embedding"` — operates on 128-d vectors, NOT images.
**What it does**: Add Laplace noise to each dimension of the 128-d embedding. Noise scale = `sensitivity / epsilon`. Smaller ε = more noise = stronger privacy. Reference notebook cells 9–22 for the old raw-pixel Laplace (adapt to embeddings).

**Class signature to implement**:

```python
import numpy as np
from .base import PPTLayer

class DifferentialPrivacy:
    layer: PPTLayer = "embedding"

    def __init__(self, epsilon: float, sensitivity: float, seed: int) -> None:
        # store epsilon, sensitivity; init np.random.default_rng(seed)
        ...

    def fit(self, X: np.ndarray) -> None:
        # optional: could calibrate sensitivity = max L1 norm of rows in X
        # acceptable to leave as no-op if caller provides sensitivity
        ...

    def apply(self, X: np.ndarray) -> np.ndarray:
        # add iid Laplace noise of scale (sensitivity / epsilon) to every element
        # return array with same shape and dtype as X
        ...
```

**Tests you must write in `tests/test_dp.py`**:
- shape preservation: `apply(X).shape == X.shape`
- stochasticity: two `apply(X)` calls produce different outputs
- determinism given seed: rebuild with same seed → same outputs
- `isinstance(DifferentialPrivacy(...), PrivacyMechanism)` is True
- as ε increases (e.g., ε=1000), mean absolute difference from input → near zero

**Paper link**: Ziller et al. 2021 (Paper 2 — your lit-review target).

---

### İrem Damla — Block-wise Image Encryption (BIE)

**Create**: `privacy_ml/ppt/bie.py`
**Layer**: `"image"` — operates on 150×150×1 images, BEFORE the encoder. Encoder will be retrained on BIE'd images by Egehan's runner (don't worry about that).
**What it does**: Divide each image into a grid of tiles, shuffle them by a key-derived permutation, return the same-shape image. Matches Kiya et al. 2023.

**Class signature to implement**:

```python
import numpy as np
from .base import PPTLayer

class BlockWiseImageEncryption:
    layer: PPTLayer = "image"

    def __init__(self, tile_size: int, key_seed: int) -> None:
        # validate: 150 % tile_size == 0 (else raise ValueError)
        # derive permutation from key_seed using np.random.default_rng(key_seed)
        # for tile_size=10: grid is 15×15 = 225 tiles
        ...

    def fit(self, X: np.ndarray) -> None:
        ...   # no calibration needed; no-op

    def apply(self, X: np.ndarray) -> np.ndarray:
        # X shape: (N, 150, 150, 1)
        # 1. reshape each image into (15, 15, tile_size, tile_size, 1)
        # 2. flatten tile index, permute by self.permutation, reshape back
        # 3. reassemble into (N, 150, 150, 1)
        # return same shape and dtype as X
        ...
```

**Tests you must write in `tests/test_bie.py`**:
- shape preservation: `apply(X).shape == X.shape`
- deterministic given key_seed; different key_seed → different permutation
- `tile_size` that doesn't divide 150 evenly → `ValueError` at construction
- pixel count preserved: `set(apply(X).flatten().tolist()) == set(X.flatten().tolist())` up to ordering
- `isinstance(BlockWiseImageEncryption(...), PrivacyMechanism)` is True

**Paper link**: Kiya et al. 2023 (Paper 1 — your lit-review target). Note: the paper uses ViT; we adapt the idea to our CNN.

---

### Eray İşçi — Secure Multi-Party Computation (SMPC)

**Create**: `privacy_ml/ppt/smpc.py`
**Layer**: `"embedding"` — operates on 128-d vectors.
**What it does**: Simulate additive secret-sharing. Split each 128-d embedding into `n_shares` random additive shares that sum back to the original. For our MIA pipeline, `apply` does share-and-reconstruct (since we're simulating overhead, not deploying to multiple physical servers). In the honest setting, reconstruction is lossless to float precision.

**Class signature to implement**:

```python
import numpy as np
from .base import PPTLayer

class SecretShareSMPC:
    layer: PPTLayer = "embedding"

    def __init__(self, n_shares: int, seed: int) -> None:
        # n_shares=2 for our 2-server setup (spec §2)
        # init np.random.default_rng(seed)
        ...

    def fit(self, X: np.ndarray) -> None:
        ...   # no-op

    def apply(self, X: np.ndarray) -> np.ndarray:
        # for each row x in X:
        #   generate (n_shares - 1) random vectors of same shape as x
        #   final share = x - sum(random shares)
        #   reconstructed = sum(all shares)  (should equal x up to fp error)
        # return shape == input shape, dtype float32
        ...
```

**Tests you must write in `tests/test_smpc.py`**:
- shape preservation
- lossless reconstruction: `np.allclose(apply(X), X, atol=1e-5)`
- deterministic given seed; different seed → different intermediate shares (even though reconstruction matches)
- `isinstance(SecretShareSMPC(...), PrivacyMechanism)` is True
- `n_shares < 2` → `ValueError`

**Paper link**: Jarin & Eshete 2021 "PRICURE" (that's Egehan's Paper 3, but it's the most relevant to SMPC; ask Egehan for notes).

---

### İlmay — Reconstruction Attack

Your notebook `reconstruction_attack.ipynb` is already using `privacy_ml` correctly. Two upgrades available now that Step 2 has landed:

1. **Auto-resolve the Kaggle nested layout**: you no longer need to hard-code the path. Use:
   ```python
   from privacy_ml.data import load_kaggle_pool
   X, y = load_kaggle_pool(Path('/content/chest_xray'), img_size=150)
   ```
   The outer or nested path both work now.

2. **Load Egehan's trained encoder from cache** (once he ships the runner, Step 3). Use the same `bie_tile_size` / `bie_key_seed` as `BlockWiseImageEncryption` when BIE is on (`bie_tile_size=0` when BIE is off).
   ```python
   from pathlib import Path
   import tensorflow as tf
   from privacy_ml.cache import cache_paths, encoder_hash

   hash_id = encoder_hash(bie_on=False, bie_key_seed=0, bie_tile_size=0, training_seed=42, epochs=10)
   paths = cache_paths(Path('results/cache'), hash_id)
   encoder = tf.keras.models.load_model(paths.encoder_weights)
   ```
   This lets you attack the *same* encoder Egehan's MIA attacks — consistent victim across both threat models.

**Your decoder**: free architecture choice. `Dense(128) → Dense(256) → Dense(22500) → reshape(150,150,1)` is fine, or a deconvolutional setup.

**Report** (paper / results table — not design spec §8, which is MIA JSONL): MSE, PSNR, SSIM, LPIPS, FID between reconstructed and original X-rays. Log per-config (baseline, DP, BIE, SMPC, combos). Slide 29 aligns with the first three; add LPIPS + FID for perceptual quality.

---

## Lit-review writing (parallel, unblocked by code)

Everyone contributes a ~300–400 word subsection to `conference_101719.tex`, after the existing Literature Review opener (line 86):

- **Eray** — Paper 1 (Kiya et al. 2023, BIE for ViT)
- **Onur** — Paper 2 (Ziller et al. 2021, DP for medical imaging)
- **Egehan** — Paper 3 (Jarin & Eshete 2021, PRICURE)
- **İrem Damla + İlmay** — free to help others or draft Methods/Results scaffolding

Use `\cite{bN}` with entries added to the `\begin{thebibliography}{00}` block at the bottom of the .tex.

## Submission checklist

1. `git pull --rebase origin main`
2. Your file in `privacy_ml/ppt/<name>.py`
3. Your tests in `tests/test_<name>.py` (≥5 tests covering the bullets above)
4. `pytest tests/ -v` — every test passes (not just yours)
5. `git add privacy_ml/ tests/ && git commit -m "Implement <name>"`
6. `git push origin main` — rebase on conflict, we keep main linear

## Pitfalls (do NOT do these)

- ❌ Modify `privacy_ml/models.py`, `privacy_ml/data.py`, `privacy_ml/cache.py`, `privacy_ml/schema.py`, or anything in `privacy_ml/ppt/base.py` / `stubs.py`. Those are Egehan's shared scaffolding.
- ❌ Add default parameter values. Global coding rule: every parameter explicit at every call site.
- ❌ Use ResNet18 or any pretrained backbone. Backbone is fixed.
- ❌ Change the embedding dimension from 128.
- ❌ Suppress a failing test with `@pytest.mark.skip` or `@pytest.mark.xfail`. Fix the code.
- ❌ Silently catch exceptions. Raise explicit error types.
- ❌ Add default values for your class constructor params — explicit at call site.
- ❌ Push if `pytest tests/` is red.

If the spec feels ambiguous, ping Egehan in chat before guessing.

## Team chat kickoff message (Turkish)

> **Ekip bilgilendirme — tasarım dokümanı ve backbone değişikliği**
>
> Repo'da `main` branch'ında tasarım dokümanı ve kod iskeleti hazır:
>
> - Tasarım: `docs/superpowers/specs/2026-04-18-mia-design.md`
> - Yol haritası (rolünüzün ne olduğu bu dosyada): `remaining_roadmap.md`
> - Kod: `privacy_ml/` klasörü altında; PPT Protocol ve kimlik stub'ları hazır (`privacy_ml/ppt/`)
>
> **Önemli — backbone sabit**: sunumun 3. slaytında "pretrained CNN backbone" yazıyor ama biz hız için notebook'taki **custom 3-conv CNN**'i (pretrained değil, sıfırdan eğitiliyor) kullanıyoruz. Sebepler tasarım dokümanının §2'sinde. **ResNet18 kullanılmıyor.** Slayt metnini güncelleyeceğiz.
>
> **Embedding boyutu**: 128-d (Dense(128, relu) çıktısı). DP ve SMPC bu 128-d vektör üzerinde çalışır. BIE ise görüntüde (150×150) tile-shuffle yapar — Kiya et al. paper'ına uygun.
>
> **Modülünüzü yazmaya başlayabilirsiniz**: `remaining_roadmap.md` dosyasında kendi bloğunuzu okuyun (Onur=DP, İrem Damla=BIE, Eray=SMPC, İlmay=reconstruction). Her blok: hangi dosyayı oluşturacağınız, class signature, zorunlu testler, referans paper. AI kodlama ajanınıza besleyebilirsiniz, gereksiz soru sormayacak kadar spesifik yazdım.
>
> **Literatür özeti**: `conference_101719.tex` içinde `\cite{bN}` stili, her paper için ~300–400 kelime.
>
> Sorular varsa yazın.
