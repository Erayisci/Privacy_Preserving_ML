"""Block-wise image encryption (BIE) — tile shuffle before the encoder.

Tiles within each 150×150 image are permuted according to a key-derived
permutation. Pixel multiset is preserved; spatial layout is not.

See remaining_roadmap.md (İrem Damla) and Kiya et al. 2023.
"""
from __future__ import annotations

import numpy as np

from .base import PPTLayer

_IMG_SIDE: int = 150


class BlockWiseImageEncryption:
    """Shuffle fixed-size spatial tiles using a deterministic permutation."""

    layer: PPTLayer = "image"

    def __init__(self, tile_size: int, key_seed: int) -> None:
        if _IMG_SIDE % tile_size != 0:
            raise ValueError(
                f"tile_size={tile_size} must divide image side {_IMG_SIDE}"
            )
        self.tile_size: int = tile_size
        self.key_seed: int = key_seed
        n_tiles = (_IMG_SIDE // tile_size) ** 2
        rng = np.random.default_rng(key_seed)
        self._perm: np.ndarray = rng.permutation(n_tiles)

    def fit(self, X: np.ndarray) -> None:
        return None

    def apply(self, X: np.ndarray) -> np.ndarray:
        if X.ndim != 4:
            raise ValueError(f"Expected X.ndim == 4, got {X.ndim}")
        n, h, w, c = X.shape
        if h != _IMG_SIDE or w != _IMG_SIDE:
            raise ValueError(
                f"Expected spatial shape ({_IMG_SIDE}, {_IMG_SIDE}), got ({h}, {w})"
            )
        ts = self.tile_size
        gh, gw = h // ts, w // ts
        if gh * gw != len(self._perm):
            raise ValueError(
                "Tile grid size does not match permutation length; "
                f"expected {len(self._perm)} tiles, got {gh * gw}"
            )
        out = np.empty_like(X)
        for i in range(n):
            out[i] = self._permute_single(X[i])
        return out

    def _permute_single(self, image: np.ndarray) -> np.ndarray:
        ts = self.tile_size
        gh, gw = _IMG_SIDE // ts, _IMG_SIDE // ts
        blocks: list[np.ndarray] = []
        for ii in range(gh):
            for jj in range(gw):
                sl = image[
                    ii * ts : (ii + 1) * ts,
                    jj * ts : (jj + 1) * ts,
                    :,
                ]
                blocks.append(np.copy(sl))
        reordered: list[np.ndarray] = [blocks[int(self._perm[k])] for k in range(len(blocks))]
        out = np.zeros_like(image)
        idx = 0
        for ii in range(gh):
            for jj in range(gw):
                out[
                    ii * ts : (ii + 1) * ts,
                    jj * ts : (jj + 1) * ts,
                    :,
                ] = reordered[idx]
                idx += 1
        return out
