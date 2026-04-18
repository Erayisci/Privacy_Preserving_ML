"""Reconstruction attack — decoder that maps 128-d embeddings back to images.

Threat model (spec §6, slide 22): an attacker obtains embeddings leaked from
the cloud and attempts to recover the original chest X-ray. They train a
decoder on (embedding, original-image) pairs from their own queries, then
apply it to a leaked embedding at evaluation time.

Module API:
  - build_decoder(...)        -> tf.keras.Model (128-d embedding -> 150x150x1 image)
  - train_decoder(...)        fits a decoder against (embeddings, images) pairs
  - compute_reconstruction_metrics(...) -> ReconstructionResult(mse, psnr, ssim)

Metrics per spec §8 / slide 29:
  - MSE  (higher = better defense)
  - PSNR (lower  = better defense; dB)
  - SSIM (closer-to-zero = better defense; structural similarity)

LPIPS and FID are out of scope for this module (heavy TF/PyTorch deps);
İlmay's notebook covers those separately.

Ported from reconstruction_attack.ipynb cells 9-12.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

DECODER_INITIAL_HW: int = 7
DECODER_INITIAL_CHANNELS: int = 128
DECODER_UPSAMPLE_FILTERS: tuple = (128, 64, 32, 16)
PSNR_DATA_RANGE: float = 1.0


class ReconstructionResult(NamedTuple):
    """Per-config reconstruction metrics, averaged across the eval set."""

    mse: float
    psnr: float
    ssim: float


def build_decoder(
    embedding_dim: int,
    img_size: int,
    channels: int,
    name: str,
) -> "tf.keras.Model":  # type: ignore[name-defined]
    """128-d embedding -> (img_size, img_size, channels) image.

    Architecture (from reconstruction_attack.ipynb cell 9):
        Dense(7*7*128) -> Reshape(7,7,128)
        -> 4x Conv2DTranspose (stride 2): 7->14->28->56->112
        -> Resize to img_size
        -> Conv2D(channels, sigmoid)
    """
    from tensorflow.keras import Model, Input, layers

    inputs = Input(shape=(embedding_dim,), name=f"{name}_input")
    x = layers.Dense(
        DECODER_INITIAL_HW * DECODER_INITIAL_HW * DECODER_INITIAL_CHANNELS,
        activation="relu",
    )(inputs)
    x = layers.Reshape(
        (DECODER_INITIAL_HW, DECODER_INITIAL_HW, DECODER_INITIAL_CHANNELS)
    )(x)
    for filters in DECODER_UPSAMPLE_FILTERS:
        x = layers.Conv2DTranspose(
            filters,
            (3, 3),
            strides=2,
            padding="same",
            activation="relu",
        )(x)
    x = layers.Resizing(img_size, img_size)(x)
    outputs = layers.Conv2D(
        channels, (3, 3), padding="same", activation="sigmoid"
    )(x)
    return Model(inputs=inputs, outputs=outputs, name=name)


def train_decoder(
    decoder: "tf.keras.Model",  # type: ignore[name-defined]
    embeddings: np.ndarray,
    images: np.ndarray,
    epochs: int,
    batch_size: int,
    seed: int,
) -> None:
    """Fit a decoder to reconstruct images from embeddings.

    The decoder learns the inverse of the (possibly-privatized) encoder.
    Higher-quality fits produce better attacks — so low MSE at eval time
    means privacy is leaking.
    """
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    decoder.compile(optimizer="adam", loss="mse")
    decoder.fit(
        embeddings,
        images,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
    )


def compute_reconstruction_metrics(
    originals: np.ndarray,
    reconstructed: np.ndarray,
) -> ReconstructionResult:
    """Per-sample MSE/PSNR/SSIM, averaged across the batch.

    Parameters
    ----------
    originals : np.ndarray of shape (N, H, W, 1), float in [0, 1]
    reconstructed : np.ndarray of same shape and dtype

    Returns
    -------
    ReconstructionResult with mean MSE, PSNR (dB), and SSIM.

    Matches reconstruction_attack.ipynb cell 12's `compute_metrics`.
    """
    from skimage.metrics import (
        peak_signal_noise_ratio as _psnr,
        structural_similarity as _ssim,
    )

    if originals.shape != reconstructed.shape:
        raise ValueError(
            f"Shape mismatch: originals {originals.shape} vs "
            f"reconstructed {reconstructed.shape}"
        )
    if originals.ndim != 4 or originals.shape[-1] != 1:
        raise ValueError(
            f"Expected (N, H, W, 1) grayscale, got {originals.shape}"
        )

    mse_vals = []
    psnr_vals = []
    ssim_vals = []
    for orig, recon in zip(originals, reconstructed):
        o = orig[:, :, 0].astype(np.float64)
        r = recon[:, :, 0].astype(np.float64)
        mse_vals.append(float(np.mean((o - r) ** 2)))
        psnr_vals.append(
            float(_psnr(o, r, data_range=PSNR_DATA_RANGE))
        )
        ssim_vals.append(
            float(_ssim(o, r, data_range=PSNR_DATA_RANGE))
        )
    return ReconstructionResult(
        mse=float(np.mean(mse_vals)),
        psnr=float(np.mean(psnr_vals)),
        ssim=float(np.mean(ssim_vals)),
    )
