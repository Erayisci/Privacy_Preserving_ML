"""Dataset loading and stratified splitting.

The Kaggle chest-xray-pneumonia pool (~5856 images) is loaded, then
reshuffled and split into canonical subsets with fixed seed=42:

  victim_members:    2000 images (in the victim's training set)
  victim_nonmembers: 1000 images (held out; "not in training" for MIA)
  shadow_pool:       2856 images (used to train Shokri shadow models)

Shadow models bootstrap-sample (train, holdout) pairs from shadow_pool;
train and holdout are disjoint within a shadow but may overlap across
shadows (standard Shokri et al. 2017 shadow sampling).

See docs/superpowers/specs/2026-04-18-mia-design.md §4.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, NamedTuple, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

VICTIM_MEMBERS_SIZE: int = 2000
VICTIM_NONMEMBERS_SIZE: int = 1000
SHADOW_POOL_SIZE: int = 2856
EXPECTED_TOTAL: int = (
    VICTIM_MEMBERS_SIZE + VICTIM_NONMEMBERS_SIZE + SHADOW_POOL_SIZE
)

NORMAL_LABEL: int = 0
PNEUMONIA_LABEL: int = 1

KAGGLE_SUBDIRS: Tuple[str, ...] = ("train", "val", "test")
KAGGLE_CLASS_DIRS: Tuple[Tuple[str, int], ...] = (
    ("NORMAL", NORMAL_LABEL),
    ("PNEUMONIA", PNEUMONIA_LABEL),
)
SUPPORTED_IMAGE_SUFFIXES: Tuple[str, ...] = (".png", ".jpg", ".jpeg")


class SplitIndices(NamedTuple):
    """Indices into the pooled (X, y) arrays for each canonical split."""

    victim_members: np.ndarray
    victim_nonmembers: np.ndarray
    shadow_pool: np.ndarray


class ShadowSplit(NamedTuple):
    """One shadow model's disjoint train/holdout index subsets."""

    train_indices: np.ndarray
    holdout_indices: np.ndarray


def class_balance(labels: np.ndarray) -> float:
    """Fraction of samples whose label equals PNEUMONIA_LABEL."""
    return float(np.mean(labels == PNEUMONIA_LABEL))


def split_pool_indices(y: np.ndarray, seed: int) -> SplitIndices:
    """Stratified 3-way split of the pool into the canonical subsets.

    Splits preserve class balance within each subset and are deterministic
    for a fixed seed. If the pool is larger than EXPECTED_TOTAL, the excess
    is discarded (stratified downsample); if smaller, a ValueError is
    raised.
    """
    total = len(y)
    if total < EXPECTED_TOTAL:
        raise ValueError(
            f"Pool has {total} samples; need at least {EXPECTED_TOTAL}"
        )

    all_indices = np.arange(total)

    shadow_indices, victim_indices = train_test_split(
        all_indices,
        train_size=SHADOW_POOL_SIZE,
        stratify=y[all_indices],
        random_state=seed,
    )

    victim_total = VICTIM_MEMBERS_SIZE + VICTIM_NONMEMBERS_SIZE
    if len(victim_indices) > victim_total:
        victim_indices, _ = train_test_split(
            victim_indices,
            train_size=victim_total,
            stratify=y[victim_indices],
            random_state=seed,
        )
    elif len(victim_indices) < victim_total:
        raise ValueError(
            f"After shadow_pool allocation, only {len(victim_indices)} "
            f"remain for victim splits; need {victim_total}"
        )

    member_indices, nonmember_indices = train_test_split(
        victim_indices,
        train_size=VICTIM_MEMBERS_SIZE,
        stratify=y[victim_indices],
        random_state=seed,
    )

    return SplitIndices(
        victim_members=member_indices,
        victim_nonmembers=nonmember_indices,
        shadow_pool=shadow_indices,
    )


def build_shadow_splits(
    shadow_indices: np.ndarray,
    y: np.ndarray,
    n_shadows: int,
    train_size: int,
    holdout_size: int,
    seed: int,
) -> List[ShadowSplit]:
    """Sample n_shadows disjoint-within (train, holdout) pairs from the shadow pool.

    Each pair is stratified on class. Pairs across shadows may share
    indices (bootstrap sampling). Deterministic for fixed seed.
    """
    if train_size + holdout_size > len(shadow_indices):
        raise ValueError(
            f"train_size({train_size}) + holdout_size({holdout_size}) "
            f"exceeds shadow pool size {len(shadow_indices)}"
        )

    shadow_labels = y[shadow_indices]
    splits: List[ShadowSplit] = []
    for shadow_k in range(n_shadows):
        train_idx, holdout_idx = train_test_split(
            shadow_indices,
            train_size=train_size,
            test_size=holdout_size,
            stratify=shadow_labels,
            random_state=seed + shadow_k,
        )
        splits.append(
            ShadowSplit(train_indices=train_idx, holdout_indices=holdout_idx)
        )
    return splits


def resolve_kaggle_base(base_dir: Path) -> Path:
    """Return the directory that directly contains train/val/test folders.

    The Kaggle `chest-xray-pneumonia` zip extracts to a nested
    `chest_xray/chest_xray/` layout, so callers passing the outer
    `chest_xray/` path would otherwise get FileNotFoundError. This
    helper probes both the given directory and a nested `chest_xray/`
    child, returning whichever one actually contains the Kaggle layout.
    """
    probe_subdir, (probe_class, _) = KAGGLE_SUBDIRS[0], KAGGLE_CLASS_DIRS[0]
    if (base_dir / probe_subdir / probe_class).is_dir():
        return base_dir
    nested = base_dir / "chest_xray"
    if (nested / probe_subdir / probe_class).is_dir():
        return nested
    raise FileNotFoundError(
        f"Could not find {probe_subdir}/{probe_class} under {base_dir} "
        f"or its nested chest_xray/ subdirectory"
    )


def load_kaggle_pool(
    base_dir: Path,
    img_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load every chest X-ray from a Kaggle layout into memory.

    Expects `<resolved>/{train,val,test}/{NORMAL,PNEUMONIA}/*.{png,jpg,jpeg}`.
    Auto-resolves the commonly-nested `chest_xray/chest_xray/` zip layout
    via `resolve_kaggle_base`.

    Images are converted to grayscale, resized to (img_size, img_size),
    and normalized to [0, 1] float32.

    Returns
    -------
    X : np.ndarray of shape (N, img_size, img_size, 1), dtype float32
    y : np.ndarray of shape (N,), dtype int64, values in {0, 1}
    """
    from tensorflow.keras.preprocessing.image import img_to_array, load_img

    resolved = resolve_kaggle_base(base_dir)

    pool_X: List[np.ndarray] = []
    pool_y: List[int] = []

    for subdir in KAGGLE_SUBDIRS:
        for class_name, class_label in KAGGLE_CLASS_DIRS:
            class_dir = resolved / subdir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected directory not found: {class_dir}"
                )
            for img_path in sorted(class_dir.iterdir()):
                if img_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                    continue
                img = load_img(
                    img_path,
                    color_mode="grayscale",
                    target_size=(img_size, img_size),
                )
                pool_X.append(img_to_array(img) / 255.0)
                pool_y.append(class_label)

    if not pool_X:
        raise ValueError(f"No images found under {resolved}")

    X = np.stack(pool_X).astype(np.float32)
    y = np.asarray(pool_y, dtype=np.int64)
    return X, y
