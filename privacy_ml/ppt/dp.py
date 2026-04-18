"""Differential Privacy — Laplace noise on the 128-d embedding.

Implements the PrivacyMechanism Protocol (privacy_ml.ppt.base) at the
"embedding" layer. For each forward pass, adds i.i.d. Laplace noise
with scale b = sensitivity / epsilon to every coordinate of every
embedding.

Smaller ε ⇒ larger noise ⇒ stronger privacy (but lower utility).
Classical ε-DP with L1 sensitivity — see Ziller et al. 2021 for the
medical-imaging framing.

Adapted from the Untitled3.ipynb prototype (cell 13, `add_laplace_noise`)
which applied Laplace noise to raw pixels. The split-model architecture
moves this to the 128-d ReLU embedding post-encoder — same mechanism,
different layer, no pixel-range clipping because embeddings aren't
bounded in [0, 1].

See docs/superpowers/specs/2026-04-18-mia-design.md §2 / slide 14.
"""
from __future__ import annotations

import numpy as np

from .base import PPTLayer


class DifferentialPrivacy:
    """ε-DP Laplace mechanism at the embedding layer.

    Stochastic: two calls with the same input produce different outputs.
    Deterministic given a fixed seed — the RNG is owned per-instance.
    """

    layer: PPTLayer = "embedding"

    def __init__(self, epsilon: float, sensitivity: float, seed: int) -> None:
        if epsilon <= 0.0:
            raise ValueError(f"epsilon must be positive, got {epsilon}")
        if sensitivity <= 0.0:
            raise ValueError(
                f"sensitivity must be positive, got {sensitivity}"
            )
        self.epsilon: float = float(epsilon)
        self.sensitivity: float = float(sensitivity)
        self._scale: float = self.sensitivity / self.epsilon
        self._rng: np.random.Generator = np.random.default_rng(seed)

    def fit(self, X: np.ndarray) -> None:
        return None

    def apply(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 2:
            raise ValueError(
                f"Expected X.ndim == 2 (N, D), got {X.ndim} with shape {X.shape}"
            )
        X_f32 = np.asarray(X, dtype=np.float32)
        noise = self._rng.laplace(
            loc=0.0, scale=self._scale, size=X_f32.shape
        ).astype(np.float32)
        return X_f32 + noise
