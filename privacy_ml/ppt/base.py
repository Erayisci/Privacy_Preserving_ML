"""The PrivacyMechanism Protocol that teammate modules must satisfy.

Teammates implement this contract in their own modules:
  - privacy_ml/ppt/dp.py
  - privacy_ml/ppt/bie.py
  - privacy_ml/ppt/smpc.py

The runner (see spec §5) reads `layer` to decide where in the pipeline
to insert each mechanism.

See docs/superpowers/specs/2026-04-18-mia-design.md §3.
"""
from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

import numpy as np

PPTLayer = Literal["image", "embedding"]


@runtime_checkable
class PrivacyMechanism(Protocol):
    """A privacy-preserving transformation with a well-defined insertion point.

    Attributes
    ----------
    layer
        Where this mechanism is inserted in the pipeline. "image"
        mechanisms transform raw images (shape `(N, H, W, C)`) before
        they reach the encoder; "embedding" mechanisms transform
        embeddings (shape `(N, D)`) between encoder and classifier head.

    Contract for `apply(X)`:
      - Input and output arrays must have identical shapes.
      - Input and output dtypes should both be `np.float32`.
      - The function should be deterministic given any randomness that
        was seeded during `fit` or construction; if the mechanism is
        itself stochastic (e.g. DP Laplace noise), different calls may
        produce different results, and the mechanism owns its RNG.

    Contract for `fit(X)`:
      - Optional calibration hook. Mechanisms that don't need calibration
        should implement this as a no-op.
      - Called before the first `apply()` with the mechanism's natural-layer
        training data (images for image-layer, embeddings for embedding-layer).
    """

    layer: PPTLayer

    def fit(self, X: np.ndarray) -> None:
        """Optional one-time calibration on training data."""
        ...

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Transform a batch. Must preserve input shape and dtype."""
        ...
