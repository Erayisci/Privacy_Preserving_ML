"""Tests for privacy_ml.ppt.dp — DifferentialPrivacy.

Per remaining_roadmap.md (Onur):
  - shape preservation
  - stochasticity (two apply calls on the same input differ)
  - determinism given seed
  - Protocol conformance (isinstance check)
  - as ε → ∞, noise → 0 (input ≈ output)
  - input validation on epsilon, sensitivity, and tensor rank
"""
from __future__ import annotations

import numpy as np
import pytest

from privacy_ml.ppt.base import PrivacyMechanism
from privacy_ml.ppt.dp import DifferentialPrivacy

_BATCH: int = 8
_EMBEDDING_DIM: int = 128
_SEED: int = 42


def _make_embeddings() -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.uniform(0.0, 1.0, size=(_BATCH, _EMBEDDING_DIM)).astype(
        np.float32
    )


def test_protocol_conformance() -> None:
    mech = DifferentialPrivacy(epsilon=1.0, sensitivity=1.0, seed=_SEED)
    assert isinstance(mech, PrivacyMechanism)
    assert mech.layer == "embedding"


def test_apply_preserves_shape_and_dtype() -> None:
    mech = DifferentialPrivacy(epsilon=1.0, sensitivity=1.0, seed=_SEED)
    X = _make_embeddings()
    Y = mech.apply(X)
    assert Y.shape == X.shape
    assert Y.dtype == np.float32


def test_apply_is_stochastic() -> None:
    mech = DifferentialPrivacy(epsilon=1.0, sensitivity=1.0, seed=_SEED)
    X = _make_embeddings()
    Y1 = mech.apply(X)
    Y2 = mech.apply(X)
    assert not np.allclose(Y1, Y2)


def test_deterministic_across_instances_with_same_seed() -> None:
    X = _make_embeddings()
    a = DifferentialPrivacy(epsilon=1.0, sensitivity=1.0, seed=_SEED)
    b = DifferentialPrivacy(epsilon=1.0, sensitivity=1.0, seed=_SEED)
    np.testing.assert_array_equal(a.apply(X), b.apply(X))


def test_different_seeds_produce_different_noise() -> None:
    X = _make_embeddings()
    a = DifferentialPrivacy(epsilon=1.0, sensitivity=1.0, seed=_SEED)
    b = DifferentialPrivacy(epsilon=1.0, sensitivity=1.0, seed=_SEED + 1)
    assert not np.allclose(a.apply(X), b.apply(X))


def test_large_epsilon_collapses_noise_toward_zero() -> None:
    mech = DifferentialPrivacy(
        epsilon=1000.0, sensitivity=1.0, seed=_SEED
    )
    X = _make_embeddings()
    Y = mech.apply(X)
    mean_abs_diff = float(np.mean(np.abs(Y - X)))
    assert mean_abs_diff < 0.01


def test_small_epsilon_injects_large_noise() -> None:
    mech = DifferentialPrivacy(epsilon=0.1, sensitivity=1.0, seed=_SEED)
    X = _make_embeddings()
    Y = mech.apply(X)
    mean_abs_diff = float(np.mean(np.abs(Y - X)))
    assert mean_abs_diff > 1.0


def test_sensitivity_linearly_scales_noise() -> None:
    X = _make_embeddings()
    low = DifferentialPrivacy(epsilon=1.0, sensitivity=1.0, seed=_SEED)
    high = DifferentialPrivacy(epsilon=1.0, sensitivity=10.0, seed=_SEED)
    low_diff = np.mean(np.abs(low.apply(X) - X))
    high_diff = np.mean(np.abs(high.apply(X) - X))
    assert high_diff > 5.0 * low_diff


def test_invalid_epsilon_raises() -> None:
    with pytest.raises(ValueError):
        DifferentialPrivacy(epsilon=0.0, sensitivity=1.0, seed=_SEED)
    with pytest.raises(ValueError):
        DifferentialPrivacy(epsilon=-1.0, sensitivity=1.0, seed=_SEED)


def test_invalid_sensitivity_raises() -> None:
    with pytest.raises(ValueError):
        DifferentialPrivacy(epsilon=1.0, sensitivity=0.0, seed=_SEED)
    with pytest.raises(ValueError):
        DifferentialPrivacy(epsilon=1.0, sensitivity=-1.0, seed=_SEED)


def test_apply_rejects_non_2d_input() -> None:
    mech = DifferentialPrivacy(epsilon=1.0, sensitivity=1.0, seed=_SEED)
    image_batch = np.zeros((4, 150, 150, 1), dtype=np.float32)
    with pytest.raises(ValueError):
        mech.apply(image_batch)


def test_fit_is_noop() -> None:
    mech = DifferentialPrivacy(epsilon=1.0, sensitivity=1.0, seed=_SEED)
    assert mech.fit(_make_embeddings()) is None
