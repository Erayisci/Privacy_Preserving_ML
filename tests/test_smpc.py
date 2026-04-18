"""Tests for SMPC secret-sharing and inference.

All tests are self-contained: no real images, no training, no Kaggle.
A randomly-initialised Keras head provides deterministic but non-trivial
weights so that numerical correctness can be verified.
"""
from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from privacy_ml.models import (
    EMBEDDING_DIM,
    DEFAULT_DROPOUT_RATE,
    build_head,
)
from privacy_ml.smpc import (
    additive_share,
    reconstruct,
    get_head_weights,
    simulate_smpc_linear,
    smpc_predict,
    smpc_accuracy,
)

_BATCH = 16
_SEED = 0
_RNG = np.random.default_rng(_SEED)


@pytest.fixture(scope="module")
def embeddings() -> np.ndarray:
    return np.random.default_rng(1).standard_normal(
        (_BATCH, EMBEDDING_DIM)
    ).astype(np.float32)


@pytest.fixture(scope="module")
def keras_head() -> tf.keras.Model:
    head = build_head(EMBEDDING_DIM, DEFAULT_DROPOUT_RATE, name="test_head")
    # Force weight initialisation with a forward pass
    _ = head(np.zeros((1, EMBEDDING_DIM), dtype=np.float32), training=False)
    return head


# ---------------------------------------------------------------------------
# Secret-sharing primitives
# ---------------------------------------------------------------------------

def test_reconstruct_is_exact(embeddings):
    rng = np.random.default_rng(10)
    s1, s2 = additive_share(embeddings, rng)
    np.testing.assert_allclose(reconstruct(s1, s2), embeddings, rtol=1e-5)


def test_shares_do_not_equal_original(embeddings):
    rng = np.random.default_rng(11)
    s1, s2 = additive_share(embeddings, rng)
    assert not np.allclose(s1, embeddings), "share1 should not equal x"
    assert not np.allclose(s2, embeddings), "share2 should not equal x"


def test_shares_are_independent_of_x(embeddings):
    """share1 ~ N(0,1) — its mean should be near 0, not near mean(x)."""
    rng = np.random.default_rng(12)
    s1, _ = additive_share(embeddings, rng)
    assert abs(s1.mean()) < 0.5, "share1 mean should be near 0"


def test_different_seeds_produce_different_shares(embeddings):
    rng_a = np.random.default_rng(20)
    rng_b = np.random.default_rng(21)
    s1_a, _ = additive_share(embeddings, rng_a)
    s1_b, _ = additive_share(embeddings, rng_b)
    assert not np.allclose(s1_a, s1_b)


def test_reconstruct_preserves_dtype(embeddings):
    rng = np.random.default_rng(13)
    s1, s2 = additive_share(embeddings, rng)
    result = reconstruct(s1, s2)
    assert result.dtype == embeddings.dtype


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------

def test_get_head_weights_shapes(keras_head):
    W, b = get_head_weights(keras_head)
    assert W.shape == (EMBEDDING_DIM, 1)
    assert b.shape == (1,)


def test_get_head_weights_raises_on_no_dense():
    flat = tf.keras.Sequential([tf.keras.layers.Flatten()])
    flat.build((None, EMBEDDING_DIM))
    with pytest.raises(ValueError, match="no Dense layer"):
        get_head_weights(flat)


# ---------------------------------------------------------------------------
# Simulated linear inference
# ---------------------------------------------------------------------------

def test_simulate_smpc_linear_matches_direct(embeddings, keras_head):
    W, b = get_head_weights(keras_head)
    rng = np.random.default_rng(30)
    s1, s2 = additive_share(embeddings, rng)

    smpc_logit = simulate_smpc_linear(s1, s2, W, b)
    direct_logit = embeddings @ W + b

    np.testing.assert_allclose(smpc_logit, direct_logit, rtol=1e-4, atol=1e-5)


def test_simulate_smpc_linear_output_shape(embeddings, keras_head):
    W, b = get_head_weights(keras_head)
    rng = np.random.default_rng(31)
    s1, s2 = additive_share(embeddings, rng)
    out = simulate_smpc_linear(s1, s2, W, b)
    assert out.shape == (_BATCH, 1)


# ---------------------------------------------------------------------------
# End-to-end smpc_predict
# ---------------------------------------------------------------------------

def test_smpc_predict_matches_keras_head(embeddings, keras_head):
    """SMPC inference must produce identical results to plain Keras inference."""
    keras_preds = keras_head(embeddings, training=False).numpy()
    smpc_preds = smpc_predict(embeddings, keras_head, seed=42)
    np.testing.assert_allclose(smpc_preds, keras_preds, rtol=1e-4, atol=1e-5)


def test_smpc_predict_output_shape(embeddings, keras_head):
    preds = smpc_predict(embeddings, keras_head)
    assert preds.shape == (_BATCH, 1)


def test_smpc_predict_output_in_unit_interval(embeddings, keras_head):
    preds = smpc_predict(embeddings, keras_head)
    assert np.all((preds >= 0.0) & (preds <= 1.0))


def test_smpc_predict_is_deterministic(embeddings, keras_head):
    p1 = smpc_predict(embeddings, keras_head, seed=99)
    p2 = smpc_predict(embeddings, keras_head, seed=99)
    np.testing.assert_array_equal(p1, p2)


def test_smpc_predict_seed_changes_shares_not_result(embeddings, keras_head):
    """Different seeds should produce the same predictions (shares cancel out)."""
    p1 = smpc_predict(embeddings, keras_head, seed=1)
    p2 = smpc_predict(embeddings, keras_head, seed=2)
    np.testing.assert_allclose(p1, p2, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# smpc_accuracy
# ---------------------------------------------------------------------------

def test_smpc_accuracy_range(embeddings, keras_head):
    preds = smpc_predict(embeddings, keras_head)
    labels = np.zeros(_BATCH, dtype=int)
    acc = smpc_accuracy(preds, labels)
    assert 0.0 <= acc <= 1.0


def test_smpc_accuracy_all_correct():
    preds = np.ones((_BATCH, 1), dtype=np.float32)
    labels = np.ones(_BATCH, dtype=int)
    assert smpc_accuracy(preds, labels) == 1.0


def test_smpc_accuracy_all_wrong():
    preds = np.ones((_BATCH, 1), dtype=np.float32)
    labels = np.zeros(_BATCH, dtype=int)
    assert smpc_accuracy(preds, labels) == 0.0


def test_smpc_accuracy_shape_mismatch_raises():
    preds = np.array([[0.7], [0.3]])
    labels = np.array([1, 0, 1])
    with pytest.raises(ValueError):
        smpc_accuracy(preds, labels)
