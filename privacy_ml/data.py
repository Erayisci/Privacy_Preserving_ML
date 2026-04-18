"""Dataset loading and Kaggle-native stratified splitting.

The Kaggle chest-xray-pneumonia dataset is loaded into a flat (X, y)
pool along with an origin annotation (load_kaggle_origins) that says
whether each image came from train/, val/, or test/. The split function
then uses those origins to carve out three subsets:

  victim_members:    2000 stratified subsample of Kaggle train/
  victim_nonmembers: all 624 of Kaggle test/
  shadow_pool:       remaining ~3216 of Kaggle train/

Kaggle val/ (only 16 images) is dropped. The train/test distribution
shift built into Kaggle's native split is what gives MIA a signal to
exploit — see spec §4 revision history.

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

# --- Canonical split sizes for the Kaggle-native partitioning ---
# Kaggle's chest-xray-pneumonia ships with train/val/test subdirs; we use
# train/ as the victim+shadow pool and test/ as victim_nonmembers so the
# natural distribution shift between train/ and test/ creates the train-test
# generalization gap that MIA exploits. val/ is only 16 images and is
# dropped. See spec §4 revision history for the retune rationale.
EXPECTED_KAGGLE_TRAIN_SIZE: int = 5216
EXPECTED_KAGGLE_TEST_SIZE: int = 624

VICTIM_MEMBERS_SIZE: int = 2000                         # subsample of Kaggle train/
VICTIM_NONMEMBERS_SIZE: int = EXPECTED_KAGGLE_TEST_SIZE # all of Kaggle test/
SHADOW_POOL_SIZE: int = EXPECTED_KAGGLE_TRAIN_SIZE - VICTIM_MEMBERS_SIZE
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


def split_pool_indices(
    y: np.ndarray,
    subdirs: np.ndarray,
    seed: int,
) -> SplitIndices:
    """Partition the pool using Kaggle's native train/test structure.

    Kaggle's test/ is known to have different radiographic characteristics
    than train/ (the dataset maintainer drew it from a distinct sampling
    protocol). Using test/ as victim_nonmembers introduces the
    distribution shift that makes the split-model classifier's
    generalization gap measurable — which in turn gives MIA a signal to
    exploit. val/ (only 16 images) is dropped.

    Parameters
    ----------
    y : np.ndarray of shape (N,), int labels {0, 1}
    subdirs : np.ndarray of shape (N,), dtype '<U5', values in
        {'train', 'val', 'test'} — the Kaggle subdir each sample was
        loaded from (see load_kaggle_origins).
    seed : int, for reproducible stratified subsampling.

    Returns
    -------
    SplitIndices with:
        victim_members: stratified VICTIM_MEMBERS_SIZE subsample of train/
        shadow_pool:    remaining train/ indices (used by Shokri shadows)
        victim_nonmembers: all test/ indices
    """
    if len(y) != len(subdirs):
        raise ValueError(
            f"y and subdirs must have matching length; "
            f"got {len(y)} and {len(subdirs)}"
        )

    train_indices = np.where(subdirs == "train")[0]
    test_indices = np.where(subdirs == "test")[0]

    if len(train_indices) < VICTIM_MEMBERS_SIZE:
        raise ValueError(
            f"Kaggle train/ has {len(train_indices)} images; need at least "
            f"{VICTIM_MEMBERS_SIZE} for victim_members"
        )
    if len(test_indices) == 0:
        raise ValueError(
            "Kaggle test/ is empty; cannot form victim_nonmembers"
        )

    member_indices, remaining_train = train_test_split(
        train_indices,
        train_size=VICTIM_MEMBERS_SIZE,
        stratify=y[train_indices],
        random_state=seed,
    )

    return SplitIndices(
        victim_members=member_indices,
        victim_nonmembers=test_indices,
        shadow_pool=remaining_train,
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


def load_kaggle_origins(base_dir: Path) -> np.ndarray:
    """Return the Kaggle subdir origin for each image in load_kaggle_pool's output.

    No image decoding — just directory enumeration. The returned array has the
    same length and ordering as the (X, y) arrays produced by load_kaggle_pool,
    so callers can zip them elementwise to know whether a sample came from
    Kaggle's train/, val/, or test/ subdirectory.

    Returns
    -------
    np.ndarray of shape (N,), dtype '<U5', values in {'train', 'val', 'test'}
    """
    resolved = resolve_kaggle_base(base_dir)
    origins: List[str] = []
    for subdir in KAGGLE_SUBDIRS:
        for class_name, _ in KAGGLE_CLASS_DIRS:
            class_dir = resolved / subdir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected directory not found: {class_dir}"
                )
            n_images = sum(
                1 for p in class_dir.iterdir()
                if p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            )
            origins.extend([subdir] * n_images)
    return np.asarray(origins, dtype="<U5")
