"""Contract tests for the PrivacyMechanism Protocol + identity stubs.

These exercise the shape-preservation and layer-declaration invariants.
They double as reference examples for teammates implementing DP/BIE/SMPC.
"""
from __future__ import annotations

import numpy as np

from privacy_ml.ppt.base import PrivacyMechanism
from privacy_ml.ppt.stubs import IdentityEmbedding, IdentityImage

_BATCH_SIZE: int = 4
_IMG_SIZE: int = 150
_CHANNELS: int = 1
_EMBEDDING_DIM: int = 128


def test_identity_embedding_satisfies_protocol() -> None:
    mech = IdentityEmbedding()
    assert isinstance(mech, PrivacyMechanism)


def test_identity_image_satisfies_protocol() -> None:
    mech = IdentityImage()
    assert isinstance(mech, PrivacyMechanism)


def test_identity_embedding_declares_embedding_layer() -> None:
    assert IdentityEmbedding.layer == "embedding"


def test_identity_image_declares_image_layer() -> None:
    assert IdentityImage.layer == "image"


def test_identity_embedding_apply_preserves_array() -> None:
    mech = IdentityEmbedding()
    X = np.random.rand(_BATCH_SIZE, _EMBEDDING_DIM).astype(np.float32)
    np.testing.assert_array_equal(mech.apply(X), X)


def test_identity_image_apply_preserves_array() -> None:
    mech = IdentityImage()
    X = np.random.rand(
        _BATCH_SIZE, _IMG_SIZE, _IMG_SIZE, _CHANNELS
    ).astype(np.float32)
    np.testing.assert_array_equal(mech.apply(X), X)


def test_identity_embedding_apply_preserves_shape() -> None:
    mech = IdentityEmbedding()
    X = np.random.rand(_BATCH_SIZE, _EMBEDDING_DIM).astype(np.float32)
    Y = mech.apply(X)
    assert Y.shape == X.shape


def test_identity_image_apply_preserves_shape() -> None:
    mech = IdentityImage()
    X = np.random.rand(
        _BATCH_SIZE, _IMG_SIZE, _IMG_SIZE, _CHANNELS
    ).astype(np.float32)
    Y = mech.apply(X)
    assert Y.shape == X.shape


def test_identity_fit_is_noop_on_embeddings() -> None:
    mech = IdentityEmbedding()
    mech.fit(np.random.rand(_BATCH_SIZE, _EMBEDDING_DIM).astype(np.float32))


def test_identity_fit_is_noop_on_images() -> None:
    mech = IdentityImage()
    mech.fit(
        np.random.rand(
            _BATCH_SIZE, _IMG_SIZE, _IMG_SIZE, _CHANNELS
        ).astype(np.float32)
    )
