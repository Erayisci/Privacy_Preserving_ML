"""SMPC private inference via 2-party additive secret sharing.

Threat model: semi-honest (honest-but-curious), static corruption,
polynomial-time adversary — matches the presentation slides.

Protocol
--------
1. Hospital encoder produces a 128-d embedding  x.
2. Hospital splits x into two additive shares:
       x  =  x1 + x2      (x1 ~ N(0,1), x2 = x - x1)
3. Server A receives x1 only; Server B receives x2 only.
4. Each server evaluates the linear head on its own share:
       Server A:  partial_A = x1 @ W
       Server B:  partial_B = x2 @ W + b
5. Partial results are summed to recover the logit without
   any server observing the raw embedding:
       logit = partial_A + partial_B  =  x @ W + b
6. sigmoid(logit) gives the final diagnosis probability.

Security guarantee: each individual share is drawn from a
distribution independent of x, so a semi-honest server holding
only one share learns nothing about the original embedding.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import tensorflow as tf

__all__ = [
    "additive_share",
    "reconstruct",
    "simulate_smpc_linear",
    "smpc_predict",
    "smpc_accuracy",
]


# ---------------------------------------------------------------------------
# Secret-sharing primitives
# ---------------------------------------------------------------------------

def additive_share(
    x: np.ndarray,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split x into two additive shares that sum back to x exactly.

    share1 is drawn from N(0, 1); share2 = x - share1.
    Each share individually is uniformly random and reveals nothing about x.
    """
    share1 = rng.standard_normal(x.shape).astype(x.dtype)
    share2 = x - share1
    return share1, share2


def reconstruct(share1: np.ndarray, share2: np.ndarray) -> np.ndarray:
    """Reconstruct x from two additive shares."""
    return share1 + share2


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------

def get_head_weights(
    keras_head: tf.keras.Model,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract (W, b) from the Dense(1) layer of the Keras classifier head.

    Returns
    -------
    W : np.ndarray, shape (embedding_dim, 1)
    b : np.ndarray, shape (1,)
    """
    dense_layers = [
        layer for layer in keras_head.layers
        if isinstance(layer, tf.keras.layers.Dense)
    ]
    if not dense_layers:
        raise ValueError("keras_head has no Dense layer")
    W, b = dense_layers[-1].get_weights()
    return W, b


# ---------------------------------------------------------------------------
# 2-party simulated inference
# ---------------------------------------------------------------------------

def simulate_smpc_linear(
    share1: np.ndarray,
    share2: np.ndarray,
    W: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """Simulate 2-party collaborative linear inference on secret shares.

    Neither server sees the full embedding; each computes only on its share:
        Server A:  partial_A = share1 @ W
        Server B:  partial_B = share2 @ W + b
    Reconstruction gives:  logit = partial_A + partial_B = x @ W + b

    Parameters
    ----------
    share1, share2 : np.ndarray, shape (N, embedding_dim)
    W              : np.ndarray, shape (embedding_dim, 1)
    b              : np.ndarray, shape (1,)

    Returns
    -------
    logit : np.ndarray, shape (N, 1)
    """
    partial_a = share1 @ W        # Server A
    partial_b = share2 @ W + b    # Server B
    return partial_a + partial_b  # Reconstruction


# ---------------------------------------------------------------------------
# End-to-end SMPC prediction
# ---------------------------------------------------------------------------

def smpc_predict(
    embeddings: np.ndarray,
    keras_head: tf.keras.Model,
    seed: int = 42,
) -> np.ndarray:
    """Run SMPC inference on a batch of embeddings.

    Splits embeddings into additive shares, runs the linear head on
    each share independently (simulating two servers), reconstructs
    the logit, and applies sigmoid.

    Parameters
    ----------
    embeddings : np.ndarray, shape (N, embedding_dim)
    keras_head : Keras Model returned by build_head()
    seed       : RNG seed for reproducible share generation

    Returns
    -------
    probabilities : np.ndarray, shape (N, 1), values in [0, 1]
    """
    W, b = get_head_weights(keras_head)
    rng = np.random.default_rng(seed)
    share1, share2 = additive_share(embeddings, rng)
    logit = simulate_smpc_linear(share1, share2, W, b)
    return _sigmoid(logit)


def smpc_accuracy(
    predictions: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Binary classification accuracy from SMPC predictions.

    Parameters
    ----------
    predictions : np.ndarray, shape (N, 1)  sigmoid probabilities
    labels      : np.ndarray, shape (N,)    ground-truth 0/1 labels
    threshold   : decision boundary

    Returns
    -------
    accuracy : float in [0, 1]
    """
    predicted = (predictions.squeeze() >= threshold).astype(int)
    return float(np.mean(predicted == labels))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))
