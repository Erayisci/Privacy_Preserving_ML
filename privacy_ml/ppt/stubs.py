"""No-op PrivacyMechanism stubs.

Used for:
  1. Smoke-testing the runner end-to-end before teammates ship real PPTs.
  2. The baseline "no defense" configuration in the 8-config results table.

Stubs preserve their input exactly; they do not add noise, permute, or
split. They are the identity transform at their declared layer.
"""
from __future__ import annotations

import numpy as np

from .base import PPTLayer


class IdentityEmbedding:
    """Identity pass-through at the embedding layer."""

    layer: PPTLayer = "embedding"

    def fit(self, X: np.ndarray) -> None:
        return None

    def apply(self, X: np.ndarray) -> np.ndarray:
        return X


class IdentityImage:
    """Identity pass-through at the image layer."""

    layer: PPTLayer = "image"

    def fit(self, X: np.ndarray) -> None:
        return None

    def apply(self, X: np.ndarray) -> np.ndarray:
        return X
