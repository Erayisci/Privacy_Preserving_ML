"""Invariant tests for the encoder + head + pipeline models.

These tests target the architecture contract from the spec, not the
notebook's implementation details — i.e. they should pass as long as
the encoder still outputs (None, 128) regardless of internal layer
reordering.
"""
from __future__ import annotations

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from privacy_ml.models import (  # noqa: E402
    DEFAULT_CHANNELS,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_IMG_SIZE,
    DEFAULT_LEARNING_RATE,
    EMBEDDING_DIM,
    SCHEMA_VERSION,
    build_encoder,
    build_end_to_end,
    build_head,
    compile_for_binary_classification,
)

_BATCH_SIZE_FOR_TEST: int = 4


def test_schema_version_is_positive_int() -> None:
    assert isinstance(SCHEMA_VERSION, int)
    assert SCHEMA_VERSION >= 1


def test_encoder_input_and_output_shapes() -> None:
    encoder = build_encoder(
        img_size=DEFAULT_IMG_SIZE,
        channels=DEFAULT_CHANNELS,
        embedding_dim=EMBEDDING_DIM,
        name="enc",
    )
    assert encoder.input_shape == (
        None,
        DEFAULT_IMG_SIZE,
        DEFAULT_IMG_SIZE,
        DEFAULT_CHANNELS,
    )
    assert encoder.output_shape == (None, EMBEDDING_DIM)


def test_head_input_and_output_shapes() -> None:
    head = build_head(
        embedding_dim=EMBEDDING_DIM,
        dropout_rate=DEFAULT_DROPOUT_RATE,
        name="head",
    )
    assert head.input_shape == (None, EMBEDDING_DIM)
    assert head.output_shape == (None, 1)


def test_encoder_output_is_nonnegative_relu() -> None:
    encoder = build_encoder(
        DEFAULT_IMG_SIZE, DEFAULT_CHANNELS, EMBEDDING_DIM, "enc"
    )
    x = np.random.rand(
        _BATCH_SIZE_FOR_TEST,
        DEFAULT_IMG_SIZE,
        DEFAULT_IMG_SIZE,
        DEFAULT_CHANNELS,
    ).astype(np.float32)
    embeddings = encoder.predict(x, verbose=0)
    assert embeddings.shape == (_BATCH_SIZE_FOR_TEST, EMBEDDING_DIM)
    assert np.all(embeddings >= 0.0)


def test_end_to_end_forward_pass_is_in_unit_interval() -> None:
    encoder = build_encoder(
        DEFAULT_IMG_SIZE, DEFAULT_CHANNELS, EMBEDDING_DIM, "enc"
    )
    head = build_head(EMBEDDING_DIM, DEFAULT_DROPOUT_RATE, "head")
    model = build_end_to_end(encoder, head, "pipeline")

    x = np.random.rand(
        _BATCH_SIZE_FOR_TEST,
        DEFAULT_IMG_SIZE,
        DEFAULT_IMG_SIZE,
        DEFAULT_CHANNELS,
    ).astype(np.float32)
    predictions = model.predict(x, verbose=0)

    assert predictions.shape == (_BATCH_SIZE_FOR_TEST, 1)
    assert np.all((predictions >= 0.0) & (predictions <= 1.0))


def test_end_to_end_shares_weights_with_components() -> None:
    encoder = build_encoder(
        DEFAULT_IMG_SIZE, DEFAULT_CHANNELS, EMBEDDING_DIM, "enc"
    )
    head = build_head(EMBEDDING_DIM, DEFAULT_DROPOUT_RATE, "head")
    pipeline = build_end_to_end(encoder, head, "pipeline")

    encoder_params = sum(w.numpy().size for w in encoder.trainable_variables)
    head_params = sum(w.numpy().size for w in head.trainable_variables)
    pipeline_params = sum(w.numpy().size for w in pipeline.trainable_variables)

    assert pipeline_params == encoder_params + head_params


def test_compile_attaches_optimizer_and_loss() -> None:
    encoder = build_encoder(
        DEFAULT_IMG_SIZE, DEFAULT_CHANNELS, EMBEDDING_DIM, "enc"
    )
    head = build_head(EMBEDDING_DIM, DEFAULT_DROPOUT_RATE, "head")
    pipeline = build_end_to_end(encoder, head, "pipeline")

    compiled = compile_for_binary_classification(
        pipeline, learning_rate=DEFAULT_LEARNING_RATE
    )
    assert compiled is pipeline
    assert pipeline.optimizer is not None
    assert pipeline.loss == "binary_crossentropy"
