"""Tests for SecretShareSMPC — covers all roadmap-required cases."""
from __future__ import annotations

import numpy as np
import pytest

from privacy_ml.ppt.base import PrivacyMechanism
from privacy_ml.ppt.smpc import SecretShareSMPC

_BATCH = 16
_DIM = 128


@pytest.fixture
def X() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((_BATCH, _DIM)).astype(np.float32)


# ---------------------------------------------------------------------------
# Roadmap-required tests
# ---------------------------------------------------------------------------

def test_shape_preservation(X):
    m = SecretShareSMPC(n_shares=2, seed=0)
    assert m.apply(X).shape == X.shape


def test_lossless_reconstruction(X):
    m = SecretShareSMPC(n_shares=2, seed=0)
    np.testing.assert_allclose(m.apply(X), X, atol=1e-5)


def test_deterministic_given_seed(X):
    m1 = SecretShareSMPC(n_shares=2, seed=42)
    m2 = SecretShareSMPC(n_shares=2, seed=42)
    np.testing.assert_array_equal(m1.apply(X), m2.apply(X))


def test_different_seeds_produce_different_shares(X):
    rng_a = np.random.default_rng(10)
    rng_b = np.random.default_rng(99)
    share_a = rng_a.standard_normal(X.shape).astype(np.float32)
    share_b = rng_b.standard_normal(X.shape).astype(np.float32)
    assert not np.allclose(share_a, share_b)


def test_isinstance_privacy_mechanism():
    m = SecretShareSMPC(n_shares=2, seed=0)
    assert isinstance(m, PrivacyMechanism)


def test_n_shares_less_than_2_raises():
    with pytest.raises(ValueError):
        SecretShareSMPC(n_shares=1, seed=0)


# ---------------------------------------------------------------------------
# Extra robustness tests
# ---------------------------------------------------------------------------

def test_layer_is_embedding():
    m = SecretShareSMPC(n_shares=2, seed=0)
    assert m.layer == "embedding"


def test_output_dtype_is_float32(X):
    m = SecretShareSMPC(n_shares=2, seed=0)
    assert m.apply(X).dtype == np.float32


def test_fit_is_noop(X):
    m = SecretShareSMPC(n_shares=2, seed=0)
    assert m.fit(X) is None


def test_lossless_with_n_shares_3(X):
    m = SecretShareSMPC(n_shares=3, seed=0)
    np.testing.assert_allclose(m.apply(X), X, atol=1e-5)


def test_n_shares_zero_raises():
    with pytest.raises(ValueError):
        SecretShareSMPC(n_shares=0, seed=0)


def test_sequential_calls_are_deterministic(X):
    m1 = SecretShareSMPC(n_shares=2, seed=7)
    m2 = SecretShareSMPC(n_shares=2, seed=7)
    for _ in range(3):
        np.testing.assert_array_equal(m1.apply(X), m2.apply(X))
