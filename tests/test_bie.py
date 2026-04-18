"""Tests for block-wise image encryption (BIE)."""
from __future__ import annotations

import numpy as np
import pytest

from privacy_ml.ppt.base import PrivacyMechanism
from privacy_ml.ppt.bie import BlockWiseImageEncryption

_TILE: int = 10
_KEY_A: int = 42
_KEY_B: int = 43
_BATCH: int = 3


def test_bie_isinstance_privacy_mechanism() -> None:
    mech = BlockWiseImageEncryption(tile_size=_TILE, key_seed=_KEY_A)
    assert isinstance(mech, PrivacyMechanism)


def test_bie_apply_preserves_shape() -> None:
    mech = BlockWiseImageEncryption(tile_size=_TILE, key_seed=_KEY_A)
    X = np.random.rand(_BATCH, 150, 150, 1).astype(np.float32)
    Y = mech.apply(X)
    assert Y.shape == X.shape


def test_bie_deterministic_per_key_seed() -> None:
    X = np.random.default_rng(0).random((2, 150, 150, 1)).astype(np.float32)
    m1 = BlockWiseImageEncryption(tile_size=_TILE, key_seed=_KEY_A)
    m2 = BlockWiseImageEncryption(tile_size=_TILE, key_seed=_KEY_A)
    np.testing.assert_array_equal(m1.apply(X), m2.apply(X))


def test_bie_different_key_seed_different_permutation() -> None:
    m_a = BlockWiseImageEncryption(tile_size=_TILE, key_seed=_KEY_A)
    m_b = BlockWiseImageEncryption(tile_size=_TILE, key_seed=_KEY_B)
    assert not np.array_equal(m_a._perm, m_b._perm)


def test_bie_invalid_tile_size_raises() -> None:
    with pytest.raises(ValueError, match="tile_size"):
        BlockWiseImageEncryption(tile_size=7, key_seed=_KEY_A)


def test_bie_pixel_values_preserved_as_set() -> None:
    mech = BlockWiseImageEncryption(tile_size=_TILE, key_seed=_KEY_A)
    X = np.random.default_rng(1).random((1, 150, 150, 1)).astype(np.float32)
    Y = mech.apply(X)
    assert set(Y.flatten().tolist()) == set(X.flatten().tolist())


def test_bie_declares_image_layer() -> None:
    mech = BlockWiseImageEncryption(tile_size=_TILE, key_seed=_KEY_A)
    assert mech.layer == "image"


def test_bie_apply_preserves_dtype() -> None:
    mech = BlockWiseImageEncryption(tile_size=_TILE, key_seed=_KEY_A)
    X = np.random.default_rng(2).random((1, 150, 150, 1)).astype(np.float32)
    Y = mech.apply(X)
    assert Y.dtype == X.dtype


def test_bie_fit_is_noop() -> None:
    mech = BlockWiseImageEncryption(tile_size=_TILE, key_seed=_KEY_A)
    X = np.random.default_rng(3).random((2, 150, 150, 1)).astype(np.float32)
    mech.fit(X)
    mech.fit(X)
    assert np.array_equal(mech.apply(X), BlockWiseImageEncryption(tile_size=_TILE, key_seed=_KEY_A).apply(X))
