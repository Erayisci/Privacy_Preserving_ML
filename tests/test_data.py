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
    EXPECTED_TOTAL,
    NORMAL_LABEL,
    PNEUMONIA_LABEL,
    SHADOW_POOL_SIZE,
    VICTIM_MEMBERS_SIZE,
    VICTIM_NONMEMBERS_SIZE,
    build_shadow_splits,
    class_balance,
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


def test_split_sizes_match_spec() -> None:
    y = _synthetic_labels(EXPECTED_TOTAL, _PNEUMONIA_FRACTION, _POOL_SEED)
    splits = split_pool_indices(y, seed=_SPLIT_SEED)
    assert len(splits.victim_members) == VICTIM_MEMBERS_SIZE
    assert len(splits.victim_nonmembers) == VICTIM_NONMEMBERS_SIZE
    assert len(splits.shadow_pool) == SHADOW_POOL_SIZE


def test_splits_are_pairwise_disjoint() -> None:
    y = _synthetic_labels(EXPECTED_TOTAL, _PNEUMONIA_FRACTION, _POOL_SEED)
    splits = split_pool_indices(y, seed=_SPLIT_SEED)
    victim_members = set(splits.victim_members.tolist())
    victim_nonmembers = set(splits.victim_nonmembers.tolist())
    shadow_pool = set(splits.shadow_pool.tolist())
    assert victim_members.isdisjoint(victim_nonmembers)
    assert victim_members.isdisjoint(shadow_pool)
    assert victim_nonmembers.isdisjoint(shadow_pool)


def test_splits_cover_exactly_expected_total() -> None:
    y = _synthetic_labels(EXPECTED_TOTAL, _PNEUMONIA_FRACTION, _POOL_SEED)
    splits = split_pool_indices(y, seed=_SPLIT_SEED)
    combined = np.concatenate(
        [splits.victim_members, splits.victim_nonmembers, splits.shadow_pool]
    )
    assert len(combined) == EXPECTED_TOTAL
    assert len(set(combined.tolist())) == EXPECTED_TOTAL


def test_class_balance_within_tolerance() -> None:
    y = _synthetic_labels(EXPECTED_TOTAL, _PNEUMONIA_FRACTION, _POOL_SEED)
    splits = split_pool_indices(y, seed=_SPLIT_SEED)
    for name, indices in (
        ("victim_members", splits.victim_members),
        ("victim_nonmembers", splits.victim_nonmembers),
        ("shadow_pool", splits.shadow_pool),
    ):
        observed = class_balance(y[indices])
        deviation = abs(observed - _PNEUMONIA_FRACTION)
        assert deviation <= _CLASS_BALANCE_TOLERANCE, (
            f"{name} balance {observed:.3f} deviates by {deviation:.3f}"
        )


def test_deterministic_with_fixed_seed() -> None:
    y = _synthetic_labels(EXPECTED_TOTAL, _PNEUMONIA_FRACTION, _POOL_SEED)
    first = split_pool_indices(y, seed=_SPLIT_SEED)
    second = split_pool_indices(y, seed=_SPLIT_SEED)
    np.testing.assert_array_equal(first.victim_members, second.victim_members)
    np.testing.assert_array_equal(
        first.victim_nonmembers, second.victim_nonmembers
    )
    np.testing.assert_array_equal(first.shadow_pool, second.shadow_pool)


def test_different_seeds_produce_different_splits() -> None:
    y = _synthetic_labels(EXPECTED_TOTAL, _PNEUMONIA_FRACTION, _POOL_SEED)
    a = split_pool_indices(y, seed=_SPLIT_SEED)
    b = split_pool_indices(y, seed=_SPLIT_SEED + 1)
    assert not np.array_equal(a.victim_members, b.victim_members)


def test_insufficient_pool_raises_value_error() -> None:
    short_pool = _synthetic_labels(
        EXPECTED_TOTAL - 100, _PNEUMONIA_FRACTION, _POOL_SEED
    )
    with pytest.raises(ValueError):
        split_pool_indices(short_pool, seed=_SPLIT_SEED)


def test_larger_pool_downsamples_to_exact_target_sizes() -> None:
    oversized = _synthetic_labels(
        EXPECTED_TOTAL + 500, _PNEUMONIA_FRACTION, _POOL_SEED
    )
    splits = split_pool_indices(oversized, seed=_SPLIT_SEED)
    assert len(splits.victim_members) == VICTIM_MEMBERS_SIZE
    assert len(splits.victim_nonmembers) == VICTIM_NONMEMBERS_SIZE
    assert len(splits.shadow_pool) == SHADOW_POOL_SIZE


def test_shadow_splits_produce_correct_count_and_sizes() -> None:
    y = _synthetic_labels(EXPECTED_TOTAL, _PNEUMONIA_FRACTION, _POOL_SEED)
    splits = split_pool_indices(y, seed=_SPLIT_SEED)
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
    y = _synthetic_labels(EXPECTED_TOTAL, _PNEUMONIA_FRACTION, _POOL_SEED)
    splits = split_pool_indices(y, seed=_SPLIT_SEED)
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
    y = _synthetic_labels(EXPECTED_TOTAL, _PNEUMONIA_FRACTION, _POOL_SEED)
    splits = split_pool_indices(y, seed=_SPLIT_SEED)
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
    y = _synthetic_labels(EXPECTED_TOTAL, _PNEUMONIA_FRACTION, _POOL_SEED)
    splits = split_pool_indices(y, seed=_SPLIT_SEED)
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
    y = _synthetic_labels(EXPECTED_TOTAL, _PNEUMONIA_FRACTION, _POOL_SEED)
    splits = split_pool_indices(y, seed=_SPLIT_SEED)
    with pytest.raises(ValueError):
        build_shadow_splits(
            shadow_indices=splits.shadow_pool,
            y=y,
            n_shadows=_N_SHADOWS_FOR_TEST,
            train_size=2000,
            holdout_size=2000,
            seed=_SPLIT_SEED,
        )
