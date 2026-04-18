"""SMPC privacy mechanism via n-party additive secret sharing.

Implements the PrivacyMechanism Protocol (privacy_ml.ppt.base).
Operates at the "embedding" layer (128-d vectors between encoder and head).

For the MIA pipeline apply() does share-then-reconstruct: it simulates
SMPC overhead without requiring physical servers. Output is numerically
identical to input (lossless for linear operations).

Threat model: semi-honest, 2-party, static corruption, polynomial-time
adversary. See docs/superpowers/specs/2026-04-18-mia-design.md §2.
"""
from __future__ import annotations

import numpy as np

from .base import PPTLayer


class SecretShareSMPC:
    """Additive secret-sharing mechanism at the embedding layer.

    Splits each 128-d embedding into n_shares random additive shares
    that sum back to the original, then reconstructs. Reconstruction
    is lossless to float32 precision.
    """

    layer: PPTLayer = "embedding"

    def __init__(self, n_shares: int, seed: int) -> None:
        if n_shares < 2:
            raise ValueError(f"n_shares must be >= 2, got {n_shares}")
        self._n_shares = n_shares
        self._rng = np.random.default_rng(seed)

    def fit(self, X: np.ndarray) -> None:
        return None

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Share-and-reconstruct each embedding row.

        Generates (n_shares - 1) random vectors; the final share is
        x minus their sum. Reconstruction sums all shares back to x.

        Parameters
        ----------
        X : np.ndarray, shape (N, D), float32

        Returns
        -------
        np.ndarray, same shape and dtype as X
        """
        X = np.asarray(X, dtype=np.float32)
        random_shares = [
            self._rng.standard_normal(X.shape).astype(np.float32)
            for _ in range(self._n_shares - 1)
        ]
        final_share = X - np.sum(random_shares, axis=0).astype(np.float32)
        all_shares = random_shares + [final_share]
        return np.sum(all_shares, axis=0).astype(np.float32)
