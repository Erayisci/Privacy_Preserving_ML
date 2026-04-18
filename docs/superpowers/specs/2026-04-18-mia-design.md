# Membership Inference Attack — Design Spec

**Project**: CS475 Privacy-Preserving ML for X-Ray Images
**Author**: Egehan Yıldız
**Date**: 2026-04-18
**Status**: Approved design; implementation pending
**Parent SSOT**: `CS475_MT_PRESENTATION.pdf` (midterm presentation deck)
**Companion artifact**: `conference_101719.tex` (IEEE conference paper)

---

## §1 — Goal

Deliver two things:

1. **MIA evaluation harness** — a CLI-driven Python package that measures how the team's three privacy-preserving technologies (DP, BIE, SMPC), individually and in combination, defend the split-model pneumonia classifier against membership inference. Output: attack accuracy + AUC for **8 PPT configurations × 2 MIA variants**, plus utility and efficiency metrics per config — enough to populate the Results and Discussion sections of the IEEE conference paper.

2. **Literature-review subsection on Jarin & Eshete 2021 ("PRICURE")** — ~300–400 words, inserted into `conference_101719.tex` after the existing Literature Review opener, formatted in the repo's IEEEtran numeric-citation style.

**Non-goals**: reconstruction attack (another teammate's task); DP/BIE/SMPC implementations (teammates'); ImageNet pretraining or architecture search.

## §2 — Architecture (SSOT: presentation slide 3)

```
                      ┌─────────── HOSPITAL-SIDE ───────────┐     CLOUD-SIDE
image (150×150×1) ──▶ [BIE?] ──▶ encoder ──▶ [DP?] ──▶ [SMPC?] ──▶ head ──▶ confidence
                      pre-enc    3-conv CNN  post-enc   post-enc       Dense(1,sigmoid)
                                 Dense(128)
```

- **Encoder**: the 3-conv CNN from `Untitled3.ipynb` cell 4, sliced at the `Dense(128, relu)` layer. Input is 150×150 grayscale; output is a 128-d embedding.
- **Head**: `Dense(1, sigmoid)` (also from the notebook's model).
- **Three PPT mechanisms (hot-swappable, teammates own implementations)**:
  - **BIE** — image-layer tile shuffle, applied *before* the encoder (matches slide 16 + reference paper 1, Kiya et al.). Encoder is *retrained on BIE'd images* when this flag is active.
  - **DP** — Laplace noise on the 128-d embedding, post-encoder (matches slide 14 mechanism diagram, adapted to embedding layer per slide 3).
  - **SMPC** — 2-server additive secret share of the 128-d embedding (matches slides 24–26).
- **Threat model for MIA (SSOT: slides 20–21)**: black-box, semi-honest external attacker. Can: submit X-ray image queries and receive prediction confidence. Cannot: access training data, model parameters, or internal activations.

**Deviation from SSOT**: slide 3 describes a "standard pretrained CNN backbone." We use the notebook's custom 3-conv CNN trained from scratch (not ImageNet-pretrained), for these reasons:

1. Speed: training completes in single-digit minutes on GPU (10–20 min on CPU).
2. MIA signal quality: the small-model + small-training-set combination deliberately overfits (notebook cell 5 training output reaches ~97% train accuracy by epoch 5; cell 6 reports ~79% test accuracy — a ~18-point generalization gap), which is the property MIA exploits. A larger pretrained model would blur this signal.
3. BIE compatibility: a trained-from-scratch encoder can be retrained on tile-shuffled images. A frozen ImageNet-pretrained backbone cannot, so image-layer BIE would break it.

The phrase "standard pretrained CNN backbone" appears only in the presentation (slide 3); the IEEE paper `conference_101719.tex` has not yet been written past the Literature Review opener (lines 85–86). When the Method section is drafted, it will describe the model as "custom 3-conv CNN trained from scratch" with a one-sentence justification aligned to the three reasons above. Any subsequent presentation will also update slide 3's phrasing.

## §3 — Package structure

No `src/` layout exists in the repo today (only `Untitled3.ipynb`, `conference_101719.tex`, `mypart.txt`, `CS475_MT_PRESENTATION.pdf`, and `README.md`). We introduce:

```
privacy_ml/                       # new top-level package
├── __init__.py
├── data.py                       # stratified splits: victim, shadow pool, MIA eval set
├── models.py                     # Encoder, Head, end-to-end pipeline assembly
├── ppt/
│   ├── __init__.py
│   ├── base.py                   # PrivacyMechanism Protocol (layer: "image" | "embedding")
│   ├── dp.py                     # teammate drops DP implementation here
│   ├── bie.py                    # teammate drops BIE implementation here
│   ├── smpc.py                   # teammate drops SMPC implementation here
│   └── stubs.py                  # no-op stubs so Egehan can test MIA pipeline before teammates ship
├── attacks/
│   ├── __init__.py
│   ├── yeom.py                   # threshold on loss/confidence (Yeom et al. 2018)
│   └── shokri.py                 # shadow-model MIA (Shokri et al. 2017)
├── metrics.py                    # utility (accuracy, F1, ECE), privacy (attack acc, AUC), efficiency
├── cache.py                      # encoder+embedding caching (see §5)
├── runner.py                     # compose pipeline from CLI flags, train, attack, log
└── run.py                        # CLI entry point: python -m privacy_ml.run ...

results/                          # created at runtime, gitignored
├── runs.jsonl                    # append-only results log; one JSON record per invocation
├── figures/                      # matplotlib plots per config
└── cache/                        # see §5

docs/superpowers/specs/           # design docs (this file)
tests/                            # pytest; see §8
```

### PPT interface contract

The contract teammates implement:

```python
# privacy_ml/ppt/base.py
from typing import Literal, Protocol
import numpy as np

class PrivacyMechanism(Protocol):
    layer: Literal["image", "embedding"]  # where this PPT inserts in the pipeline

    def fit(self, X: np.ndarray) -> None:
        """Optional calibration on training data. Default: no-op."""
        ...

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Transform a batch. X.shape = (N, 150, 150, 1) for image-layer;
        X.shape = (N, 128) for embedding-layer."""
        ...
```

The runner reads `mech.layer` and inserts the module at the correct pipeline position.

## §4 — Data flow and splits

**Source**: Kaggle `paultimothymooney/chest-xray-pneumonia` (downloaded via `kagglehub` or `kaggle` CLI). Total pool across `train/`, `val/`, `test/` directories ≈ 5856 grayscale chest X-ray images, binary label NORMAL vs PNEUMONIA. Class imbalance ~3:1 pneumonia:normal (slide 28).

**Canonical splits** (reshuffled from the Kaggle pool with `seed=42`, stratified on class):

| Split | Size | Purpose |
|---|---|---|
| `victim_members` | 2000 | Train the victim encoder + head; "in-training" set for MIA |
| `victim_nonmembers` | 1000 | Held-out from victim; "not-in-training" set for MIA |
| `shadow_pool` | 2856 | Used to train Shokri shadow models (§6) |

**MIA evaluation set** = 1000 randomly-drawn `victim_members` + 1000 `victim_nonmembers` = 2000 balanced queries per PPT config.

**Shadow model data (Shokri)**: 5 shadow models, each bootstrap-sampled from `shadow_pool` with 500 "member" + 500 "non-member" images per shadow, no overlap within a shadow but overlap across shadows is allowed. This matches Shokri et al.'s original sampling strategy and works with the finite pool size (5 × 1000 = 5000 > |shadow_pool|=2856, so disjoint-across-shadows is infeasible).

All splits have a unit test that asserts no leakage between `victim_members` and `victim_nonmembers`, and that class balance within each split is within ±2% of the global 3:1 ratio.

## §5 — Caching (critical for iteration speed)

Encoders and embeddings are cached on disk keyed by a 12-hex-digit SHA-1 of the encoder's configuration:

```
encoder_hash = sha1(
    f"{bie_on}|{bie_key_seed}|{training_seed}|{epochs}|{SCHEMA_VERSION}"
)[:12]
```

Cache layout:

```
results/cache/
├── encoders/
│   ├── {hash}.keras              # trained encoder + head weights
│   └── {hash}.meta.json          # config that produced this hash
└── embeddings/
    └── {hash}/
        ├── victim_members.npy         # shape (2000, 128), float32
        ├── victim_nonmembers.npy      # shape (1000, 128)
        ├── shadow_{k}_train.npy       # k ∈ {0..4}
        └── shadow_{k}_holdout.npy
```

**Runner startup sequence:**

1. Parse CLI flags → compute `encoder_hash`.
2. If `cache/encoders/{hash}.keras` exists, load it; else train encoder on `victim_members` (+ each shadow's member split if Shokri enabled) and save.
3. If embeddings `cache/embeddings/{hash}/` complete, memory-map the .npy files; else `encoder.predict` over each split and save.
4. Apply `--dp`/`--smpc` modules on the in-memory embedding arrays (these operate on 128-d vectors; they do not change the encoder).
5. Train the head on (possibly-privatized) embeddings.
6. Run enabled MIA variants.
7. Append one JSON record to `results/runs.jsonl`.

**Invalidation**: `SCHEMA_VERSION = 1` constant in `privacy_ml/models.py`. Bumping it forces all cache entries to be recomputed. This protects against stale caches if the encoder architecture changes.

**Expected speedup**: a DP ε sweep of 6 values reuses the same encoder and embeddings; only the head retraining (~seconds) and MIA (~seconds) runs per sweep point. First run per encoder config takes the full training time; subsequent runs take seconds.

## §6 — MIA attacks

### Yeom et al. (2018) — loss threshold

For each query `x`, compute the victim's loss `ℓ(f(x), y_true(x))`. Members typically have lower loss than non-members. Pick an optimal threshold `τ` from the loss distribution on the evaluation set; classify `ℓ(x) < τ ⇒ member`. Report attack accuracy and AUC.

Zero extra training required; implementable in <100 lines.

### Shokri et al. (2017) — shadow-model attack

1. Train 5 shadow encoders + heads on their respective bootstrap samples from `shadow_pool` (§4). These shadows mimic the victim's training behavior.
2. For each shadow, produce labeled (`confidence_vector`, `member ∈ {0,1}`) pairs from its train+holdout splits.
3. Pool all shadow pairs and train an **attack classifier** (small MLP: Dense(64, relu) → Dense(32, relu) → Dense(1, sigmoid)) to predict membership from a confidence vector.
4. Apply the attack classifier to the victim's confidence vectors on the 2000-query eval set. Report attack accuracy and AUC.

Shadow encoders inherit the same architecture as the victim and share the caching layer.

## §7 — CLI

```
python -m privacy_ml.run
    [--dp]                          # enable DP on embeddings
    [--dp-epsilon FLOAT]            # default 1.0 (inherited from notebook cell 9)
    [--bie]                         # enable image-layer BIE
    [--bie-key-seed INT]            # default 0
    [--smpc]                        # enable SMPC on embeddings
    [--smpc-shares INT]             # default 2
    [--mia yeom,shokri]             # default: both
    [--epochs INT]                  # default 10
    [--seed INT]                    # default 42
    [--output-dir PATH]             # default ./results
    [--tag STR]                     # optional human-readable label for the JSON record
    [--no-train]                    # skip encoder training if cache hit
```

**The 8-row Results table for the paper** is produced by running:

| Tag | Command |
|---|---|
| `baseline` | `python -m privacy_ml.run --tag baseline` |
| `dp-only` | `python -m privacy_ml.run --dp --tag dp-only` |
| `bie-only` | `python -m privacy_ml.run --bie --tag bie-only` |
| `smpc-only` | `python -m privacy_ml.run --smpc --tag smpc-only` |
| `dp+bie` | `python -m privacy_ml.run --dp --bie --tag dp+bie` |
| `dp+smpc` | `python -m privacy_ml.run --dp --smpc --tag dp+smpc` |
| `bie+smpc` | `python -m privacy_ml.run --bie --smpc --tag bie+smpc` |
| `all-three` | `python -m privacy_ml.run --dp --bie --smpc --tag all-three` |

Each row produces utility + privacy + efficiency metrics for both Yeom and Shokri attacks.

## §8 — Metrics (SSOT: slides 27–29)

Per invocation, emit this JSON schema to `results/runs.jsonl`:

```json
{
  "tag": "dp-only",
  "config": {"dp": true, "dp_epsilon": 1.0, "bie": false, "smpc": false, "seed": 42},
  "encoder_hash": "a1b2c3d4e5f6",
  "utility": {
    "test_accuracy": 0.76,
    "f1": 0.82,
    "ece": 0.04,
    "accuracy_drop_vs_baseline": 0.02
  },
  "privacy": {
    "yeom":   {"attack_accuracy": 0.57, "attack_auc": 0.59},
    "shokri": {"attack_accuracy": 0.54, "attack_auc": 0.56}
  },
  "efficiency": {
    "train_latency_seconds": 185.4,
    "inference_latency_ms_per_query": 1.2,
    "memory_peak_mb": 420.0,
    "embedding_bytes_per_query": 512
  },
  "timestamp": "2026-04-18T16:45:00Z"
}
```

- **Utility targets** (slide 28): accuracy drop < 5%; F1 ≥ baseline − 0.05; ECE tracked for overconfidence under noise.
- **Privacy targets** (slide 29): attack accuracy → 50% (random guessing = perfect defense); AUC → 0.5.
- **Efficiency** (slide 30): latency, memory, bandwidth; no hard targets, reported as-is.

## §9 — Testing

Per the project's CLAUDE.md ("write tests alongside new code"):

- **`tests/test_data.py`**: asserts no leakage between `victim_members` and `victim_nonmembers`; class balance within ±2% of global 3:1; `shadow_pool` is disjoint from victim splits; reshuffling with fixed seed is deterministic.
- **`tests/test_models.py`**: asserts the encoder slice exposes a 128-d output on a 150×150×1 input; head input shape is (128,); a forward pass through a stub pipeline returns a single sigmoid scalar.
- **`tests/test_ppt_stubs.py`**: asserts the stub `PrivacyMechanism` implementations conform to the Protocol and are no-op identity transforms.
- **`tests/test_attacks.py`**: on a synthetic dataset where the victim perfectly memorizes its training set, Yeom attack achieves ≥ 90% accuracy (sanity check that the attack is implemented correctly).
- **Smoke test** (`tests/test_smoke.py`): runs `python -m privacy_ml.run --tag smoke --epochs 1` on a 10-image toy dataset with all PPTs stubbed; should complete in seconds (target budget: tens of seconds, hardware-dependent) and must emit a valid JSONL record.
- **Golden test** (`tests/test_golden.py`, optional/manual): reproduces the notebook's ~79% test accuracy within ±2 points.
- **Invariant check**: under the baseline config (no PPT flags), the Yeom attack should achieve > 55% accuracy; if not, the experimental setup is broken (victim isn't actually overfitting) — this is an assertion in `runner.py`.

Minimum coverage target: 80% on new code (per global CLAUDE.md).

## §10 — Deliverables

1. The `privacy_ml/` package above, runnable end-to-end on CPU or Colab GPU.
2. `results/runs.jsonl` populated with the 8 canonical configurations (§7 table).
3. Matplotlib figures summarizing attack accuracy vs PPT config, per MIA variant.
4. ~300–400 words on PRICURE (Jarin & Eshete 2021) inserted into `conference_101719.tex` as a subsection under `\section{Literature Review}` (already stubbed at line 85). Numeric `\cite{bN}` style; bibliography entry added to the `thebibliography` environment.
5. One-line paper amendment in the Method section: "We use a custom 3-conv CNN trained from scratch on the chest X-ray pool rather than a pretrained backbone, for efficiency and to deliberately preserve the generalization gap that MIA exploits."

## §11 — Scope notes and caveats

- `mypart.txt` is gitignored (`.gitignore` = `*.txt`); team task assignments in that file are local to Egehan's workstation.
- The notebook's cell 4 shows 3339 train / 1893 val / 624 test but cell 14 shows 5216 train from the same directory. This is a path-extraction inconsistency in the Kaggle ZIP layout; the reliable total is ~5856. Our splits ignore the Kaggle subfolder structure and re-pool.
- DP, BIE, SMPC implementation ownership within the team is not formally documented in the repo. This spec assumes teammates will claim modules via `privacy_ml/ppt/{dp,bie,smpc}.py`. Ownership to be confirmed with the team before coding begins.
- The notebook's image-Laplace experiment (cells 13–22; cell 9 is unrelated dummy "segment reconstruction" code left over from an earlier iteration) applies noise to *raw pixels*, not to embeddings. This is inconsistent with the split-model architecture in slide 3 and will be replaced by the DP module operating on 128-d embeddings.
- `conference_101719.tex` currently contains IEEE template boilerplate; the final submission must strip this before submission, per the red-text warning at lines 278–280 (`\color{red}` + guidance paragraph). Out of scope for this spec.
