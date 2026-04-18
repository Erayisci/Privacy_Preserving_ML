"""Encoder + head models for the split-model pipeline.

Architecture mirrors Untitled3.ipynb cell 4, sliced at the
Dense(embedding_dim, relu) layer so the encoder produces a 128-d
embedding and the head takes embeddings to a single sigmoid.

See docs/superpowers/specs/2026-04-18-mia-design.md §2.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models

from .schema import SCHEMA_VERSION

__all__ = [
    "SCHEMA_VERSION",
    "DEFAULT_IMG_SIZE",
    "DEFAULT_CHANNELS",
    "EMBEDDING_DIM",
    "DEFAULT_DROPOUT_RATE",
    "DEFAULT_LEARNING_RATE",
    "build_encoder",
    "build_head",
    "build_end_to_end",
    "compile_for_binary_classification",
]

DEFAULT_IMG_SIZE: int = 150
DEFAULT_CHANNELS: int = 1
EMBEDDING_DIM: int = 128
DEFAULT_DROPOUT_RATE: float = 0.3
DEFAULT_LEARNING_RATE: float = 1e-3


def build_encoder(
    img_size: int,
    channels: int,
    embedding_dim: int,
    name: str,
) -> tf.keras.Model:
    """Image-to-embedding encoder.

    Input shape: (img_size, img_size, channels) grayscale image.
    Output shape: (embedding_dim,) ReLU-activated embedding.
    """
    inputs = layers.Input(
        shape=(img_size, img_size, channels),
        name=f"{name}_input",
    )
    x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    embedding = layers.Dense(
        embedding_dim,
        activation="relu",
        name=f"{name}_embedding",
    )(x)
    return models.Model(inputs=inputs, outputs=embedding, name=name)


def build_head(
    embedding_dim: int,
    dropout_rate: float,
    name: str,
) -> tf.keras.Model:
    """Classifier head on top of an embedding.

    Input shape: (embedding_dim,).
    Output shape: (1,) sigmoid probability of PNEUMONIA.
    """
    inputs = layers.Input(shape=(embedding_dim,), name=f"{name}_input")
    x = layers.Dropout(dropout_rate)(inputs)
    outputs = layers.Dense(1, activation="sigmoid", name=f"{name}_output")(x)
    return models.Model(inputs=inputs, outputs=outputs, name=name)


def build_end_to_end(
    encoder: tf.keras.Model,
    head: tf.keras.Model,
    name: str,
) -> tf.keras.Model:
    """Chain an encoder and head into a single image-to-prediction model.

    Weights are shared with the constituent models, so training this
    model trains both encoder and head in lockstep.
    """
    inputs = layers.Input(shape=encoder.input_shape[1:], name=f"{name}_input")
    embedding = encoder(inputs)
    predictions = head(embedding)
    return models.Model(inputs=inputs, outputs=predictions, name=name)


def compile_for_binary_classification(
    model: tf.keras.Model,
    learning_rate: float,
) -> tf.keras.Model:
    """Compile with Adam + binary crossentropy (matches notebook cell 4)."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
