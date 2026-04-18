"""Invariant tests for dataset splitting.

Synthesized labels (no real images) exercise the splitting logic:
- exact split sizes
- pairwise disjointness
- class balance preservation (within tolerance)
- determinism under fixed seed
- error paths on insufficient pool
- shadow-split bootstrap correctness
"""
from __future__ import annotations

import numpy as np
import pytest

from privacy_ml.data import (
    EXPECTED_KAGGLE_TEST_SIZE,
    EXPECTED_KAGGLE_TRAIN_SIZE,
    EXPECTED_TOTAL,
    KAGGLE_CLASS_DIRS,
    KAGGLE_SUBDIRS,
    NORMAL_LABEL,
    PNEUMONIA_LABEL,
    SHADOW_POOL_SIZE,
    VICTIM_MEMBERS_SIZE,
    VICTIM_NONMEMBERS_SIZE,
    build_shadow_splits,
    class_balance,
    load_kaggle_origins,
    resolve_kaggle_base,
    split_pool_indices,
)

_PNEUMONIA_FRACTION: float = 0.75
_CLASS_BALANCE_TOLERANCE: float = 0.02
_N_SHADOWS_FOR_TEST: int = 5
_SHADOW_TRAIN_SIZE: int = 500
_SHADOW_HOLDOUT_SIZE: int = 500
_SPLIT_SEED: int = 42
_POOL_SEED: int = 0


def _synthetic_labels(total: int, pneumonia_fraction: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_pneumonia = int(round(total * pneumonia_fraction))
    n_normal = total - n_pneumonia
    labels = np.concatenate(
        [
            np.full(n_pneumonia, PNEUMONIA_LABEL, dtype=np.int64),
            np.full(n_normal, NORMAL_LABEL, dtype=np.int64),
        ]
    )
    rng.shuffle(labels)
    return labels


def _synthetic_native_pool(
    n_train: int, n_test: int, pneumonia_fraction: float, seed: int
):
    """Build (y, subdirs) for a fake Kaggle-native layout."""
    rng = np.random.default_rng(seed)

    def _stratified_labels(n: int) -> np.ndarray:
        n_pneu = int(round(n * pneumonia_fraction))
        labs = np.concatenate(
            [
                np.full(n_pneu, PNEUMONIA_LABEL, dtype=np.int64),
                np.full(n - n_pneu, NORMAL_LABEL, dtype=np.int64),
            ]
        )
        rng.shuffle(labs)
        return labs

    y_train = _stratified_labels(n_train)
    y_test = _stratified_labels(n_test)
    y = np.concatenate([y_train, y_test])
    subdirs = np.concatenate(
        [
            np.full(n_train, "train", dtype="<U5"),
            np.full(n_test, "test", dtype="<U5"),
        ]
    )
    return y, subdirs


def test_native_split_sizes_match_spec() -> None:
    y, subdirs = _synthetic_native_pool(
        n_train=EXPECTED_KAGGLE_TRAIN_SIZE,
        n_test=EXPECTED_KAGGLE_TEST_SIZE,
        pneumonia_fraction=_PNEUMONIA_FRACTION,
        seed=_POOL_SEED,
    )
    splits = split_pool_indices(y, subdirs, seed=_SPLIT_SEED)
    assert len(splits.victim_members) == VICTIM_MEMBERS_SIZE
    assert len(splits.victim_nonmembers) == VICTIM_NONMEMBERS_SIZE
    assert len(splits.shadow_pool) == SHADOW_POOL_SIZE


def test_native_splits_are_pairwise_disjoint() -> None:
    y, subdirs = _synthetic_native_pool(
        n_train=EXPECTED_KAGGLE_TRAIN_SIZE,
        n_test=EXPECTED_KAGGLE_TEST_SIZE,
        pneumonia_fraction=_PNEUMONIA_FRACTION,
        seed=_POOL_SEED,
    )
    splits = split_pool_indices(y, subdirs, seed=_SPLIT_SEED)
    members = set(splits.victim_members.tolist())
    nonmembers = set(splits.victim_nonmembers.tolist())
    shadow = set(splits.shadow_pool.tolist())
    assert members.isdisjoint(nonmembers)
    assert members.isdisjoint(shadow)
    assert nonmembers.isdisjoint(shadow)


def test_native_victim_members_are_drawn_only_from_train_subdir() -> None:
    y, subdirs = _synthetic_native_pool(
        n_train=EXPECTED_KAGGLE_TRAIN_SIZE,
        n_test=EXPECTED_KAGGLE_TEST_SIZE,
        pneumonia_fraction=_PNEUMONIA_FRACTION,
        seed=_POOL_SEED,
    )
    splits = split_pool_indices(y, subdirs, seed=_SPLIT_SEED)
    assert np.all(subdirs[splits.victim_members] == "train")
    assert np.all(subdirs[splits.shadow_pool] == "train")
    assert np.all(subdirs[splits.victim_nonmembers] == "test")


def test_native_class_balance_within_tolerance() -> None:
    y, subdirs = _synthetic_native_pool(
        n_train=EXPECTED_KAGGLE_TRAIN_SIZE,
        n_test=EXPECTED_KAGGLE_TEST_SIZE,
        pneumonia_fraction=_PNEUMONIA_FRACTION,
        seed=_POOL_SEED,
    )
    splits = split_pool_indices(y, subdirs, seed=_SPLIT_SEED)
    for name, indices in (
        ("victim_members", splits.victim_members),
        ("shadow_pool", splits.shadow_pool),
    ):
        observed = class_balance(y[indices])
        deviation = abs(observed - _PNEUMONIA_FRACTION)
        assert deviation <= _CLASS_BALANCE_TOLERANCE, (
            f"{name} balance {observed:.3f} deviates by {deviation:.3f}"
        )


def test_native_split_deterministic_with_fixed_seed() -> None:
    y, subdirs = _synthetic_native_pool(
        n_train=EXPECTED_KAGGLE_TRAIN_SIZE,
        n_test=EXPECTED_KAGGLE_TEST_SIZE,
        pneumonia_fraction=_PNEUMONIA_FRACTION,
        seed=_POOL_SEED,
    )
    first = split_pool_indices(y, subdirs, seed=_SPLIT_SEED)
    second = split_pool_indices(y, subdirs, seed=_SPLIT_SEED)
    np.testing.assert_array_equal(first.victim_members, second.victim_members)
    np.testing.assert_array_equal(first.victim_nonmembers, second.victim_nonmembers)
    np.testing.assert_array_equal(first.shadow_pool, second.shadow_pool)


def test_native_split_different_seeds_differ() -> None:
    y, subdirs = _synthetic_native_pool(
        n_train=EXPECTED_KAGGLE_TRAIN_SIZE,
        n_test=EXPECTED_KAGGLE_TEST_SIZE,
        pneumonia_fraction=_PNEUMONIA_FRACTION,
        seed=_POOL_SEED,
    )
    a = split_pool_indices(y, subdirs, seed=_SPLIT_SEED)
    b = split_pool_indices(y, subdirs, seed=_SPLIT_SEED + 1)
    assert not np.array_equal(a.victim_members, b.victim_members)


def test_native_insufficient_train_pool_raises_value_error() -> None:
    y, subdirs = _synthetic_native_pool(
        n_train=VICTIM_MEMBERS_SIZE - 1,  # one short
        n_test=EXPECTED_KAGGLE_TEST_SIZE,
        pneumonia_fraction=_PNEUMONIA_FRACTION,
        seed=_POOL_SEED,
    )
    with pytest.raises(ValueError):
        split_pool_indices(y, subdirs, seed=_SPLIT_SEED)


def test_native_missing_test_subdir_raises_value_error() -> None:
    y_train = np.full(
        EXPECTED_KAGGLE_TRAIN_SIZE, PNEUMONIA_LABEL, dtype=np.int64
    )
    subdirs = np.full(
        EXPECTED_KAGGLE_TRAIN_SIZE, "train", dtype="<U5"
    )
    with pytest.raises(ValueError):
        split_pool_indices(y_train, subdirs, seed=_SPLIT_SEED)


def test_shadow_splits_produce_correct_count_and_sizes() -> None:
    y, subdirs = _synthetic_native_pool(
        n_train=EXPECTED_KAGGLE_TRAIN_SIZE,
        n_test=EXPECTED_KAGGLE_TEST_SIZE,
        pneumonia_fraction=_PNEUMONIA_FRACTION,
        seed=_POOL_SEED,
    )
    splits = split_pool_indices(y, subdirs, seed=_SPLIT_SEED)
    shadow_splits = build_shadow_splits(
        shadow_indices=splits.shadow_pool,
        y=y,
        n_shadows=_N_SHADOWS_FOR_TEST,
        train_size=_SHADOW_TRAIN_SIZE,
        holdout_size=_SHADOW_HOLDOUT_SIZE,
        seed=_SPLIT_SEED,
    )
    assert len(shadow_splits) == _N_SHADOWS_FOR_TEST
    for shadow in shadow_splits:
        assert len(shadow.train_indices) == _SHADOW_TRAIN_SIZE
        assert len(shadow.holdout_indices) == _SHADOW_HOLDOUT_SIZE


def test_shadow_train_and_holdout_disjoint_within_each_shadow() -> None:
    y, subdirs = _synthetic_native_pool(
        n_train=EXPECTED_KAGGLE_TRAIN_SIZE,
        n_test=EXPECTED_KAGGLE_TEST_SIZE,
        pneumonia_fraction=_PNEUMONIA_FRACTION,
        seed=_POOL_SEED,
    )
    splits = split_pool_indices(y, subdirs, seed=_SPLIT_SEED)
    shadow_splits = build_shadow_splits(
        shadow_indices=splits.shadow_pool,
        y=y,
        n_shadows=_N_SHADOWS_FOR_TEST,
        train_size=_SHADOW_TRAIN_SIZE,
        holdout_size=_SHADOW_HOLDOUT_SIZE,
        seed=_SPLIT_SEED,
    )
    for shadow in shadow_splits:
        train_set = set(shadow.train_indices.tolist())
        holdout_set = set(shadow.holdout_indices.tolist())
        assert train_set.isdisjoint(holdout_set)


def test_shadow_indices_are_subset_of_shadow_pool() -> None:
    y, subdirs = _synthetic_native_pool(
        n_train=EXPECTED_KAGGLE_TRAIN_SIZE,
        n_test=EXPECTED_KAGGLE_TEST_SIZE,
        pneumonia_fraction=_PNEUMONIA_FRACTION,
        seed=_POOL_SEED,
    )
    splits = split_pool_indices(y, subdirs, seed=_SPLIT_SEED)
    shadow_pool_set = set(splits.shadow_pool.tolist())
    shadow_splits = build_shadow_splits(
        shadow_indices=splits.shadow_pool,
        y=y,
        n_shadows=_N_SHADOWS_FOR_TEST,
        train_size=_SHADOW_TRAIN_SIZE,
        holdout_size=_SHADOW_HOLDOUT_SIZE,
        seed=_SPLIT_SEED,
    )
    for shadow in shadow_splits:
        assert set(shadow.train_indices.tolist()).issubset(shadow_pool_set)
        assert set(shadow.holdout_indices.tolist()).issubset(shadow_pool_set)


def test_shadow_splits_are_deterministic_with_fixed_seed() -> None:
    y, subdirs = _synthetic_native_pool(
        n_train=EXPECTED_KAGGLE_TRAIN_SIZE,
        n_test=EXPECTED_KAGGLE_TEST_SIZE,
        pneumonia_fraction=_PNEUMONIA_FRACTION,
        seed=_POOL_SEED,
    )
    splits = split_pool_indices(y, subdirs, seed=_SPLIT_SEED)
    first = build_shadow_splits(
        splits.shadow_pool,
        y,
        _N_SHADOWS_FOR_TEST,
        _SHADOW_TRAIN_SIZE,
        _SHADOW_HOLDOUT_SIZE,
        _SPLIT_SEED,
    )
    second = build_shadow_splits(
        splits.shadow_pool,
        y,
        _N_SHADOWS_FOR_TEST,
        _SHADOW_TRAIN_SIZE,
        _SHADOW_HOLDOUT_SIZE,
        _SPLIT_SEED,
    )
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a.train_indices, b.train_indices)
        np.testing.assert_array_equal(a.holdout_indices, b.holdout_indices)


def test_shadow_oversize_raises_value_error() -> None:
    y, subdirs = _synthetic_native_pool(
        n_train=EXPECTED_KAGGLE_TRAIN_SIZE,
        n_test=EXPECTED_KAGGLE_TEST_SIZE,
        pneumonia_fraction=_PNEUMONIA_FRACTION,
        seed=_POOL_SEED,
    )
    splits = split_pool_indices(y, subdirs, seed=_SPLIT_SEED)
    shadow_pool_size = len(splits.shadow_pool)
    # Pick sizes whose sum deliberately exceeds the shadow pool so the
    # constructor must raise. Using (pool, pool) guarantees overflow
    # regardless of how SHADOW_POOL_SIZE is retuned in data.py.
    with pytest.raises(ValueError):
        build_shadow_splits(
            shadow_indices=splits.shadow_pool,
            y=y,
            n_shadows=_N_SHADOWS_FOR_TEST,
            train_size=shadow_pool_size,
            holdout_size=shadow_pool_size,
            seed=_SPLIT_SEED,
        )


def _materialize_kaggle_layout(root):
    for subdir in KAGGLE_SUBDIRS:
        for class_name, _ in KAGGLE_CLASS_DIRS:
            (root / subdir / class_name).mkdir(parents=True)


def test_resolve_kaggle_base_handles_flat_layout(tmp_path):
    _materialize_kaggle_layout(tmp_path)
    assert resolve_kaggle_base(tmp_path) == tmp_path


def test_resolve_kaggle_base_handles_nested_layout(tmp_path):
    _materialize_kaggle_layout(tmp_path / "chest_xray")
    assert resolve_kaggle_base(tmp_path) == tmp_path / "chest_xray"


def test_resolve_kaggle_base_raises_when_neither_layout_present(tmp_path):
    with pytest.raises(FileNotFoundError):
        resolve_kaggle_base(tmp_path)


def test_load_kaggle_origins_matches_load_kaggle_pool_ordering(tmp_path):
    # Build a fake Kaggle layout with known per-class counts
    from PIL import Image

    counts = {
        ("train", "NORMAL"): 3,
        ("train", "PNEUMONIA"): 5,
        ("val", "NORMAL"): 1,
        ("val", "PNEUMONIA"): 1,
        ("test", "NORMAL"): 2,
        ("test", "PNEUMONIA"): 4,
    }
    for (subdir, class_name), n in counts.items():
        cls_dir = tmp_path / subdir / class_name
        cls_dir.mkdir(parents=True)
        for i in range(n):
            Image.new("L", (10, 10)).save(cls_dir / f"img_{i:03d}.png")

    from privacy_ml.data import load_kaggle_origins
    origins = load_kaggle_origins(tmp_path)

    expected = (
        ["train"] * counts[("train", "NORMAL")]
        + ["train"] * counts[("train", "PNEUMONIA")]
        + ["val"]   * counts[("val", "NORMAL")]
        + ["val"]   * counts[("val", "PNEUMONIA")]
        + ["test"]  * counts[("test", "NORMAL")]
        + ["test"]  * counts[("test", "PNEUMONIA")]
    )
    assert origins.tolist() == expected
    assert origins.shape == (sum(counts.values()),)
