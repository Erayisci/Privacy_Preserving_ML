"""Tests for privacy_ml.attacks.reconstruction.

Split into TF-free (metric math) and TF-dependent (decoder construction / forward pass).
"""
from __future__ import annotations

import numpy as np
import pytest

from privacy_ml.attacks.reconstruction import (
    ReconstructionResult,
    compute_reconstruction_metrics,
)


_BATCH: int = 4
_IMG: int = 150
_CHANNELS: int = 1
_EMBEDDING_DIM: int = 128


# --- TF-free tests: metric math ---


def test_metrics_returns_reconstruction_result() -> None:
    x = np.random.rand(_BATCH, _IMG, _IMG, _CHANNELS).astype(np.float32)
    result = compute_reconstruction_metrics(x, x)
    assert isinstance(result, ReconstructionResult)


def test_perfect_reconstruction_has_zero_mse() -> None:
    x = np.random.rand(_BATCH, _IMG, _IMG, _CHANNELS).astype(np.float32)
    result = compute_reconstruction_metrics(x, x)
    assert result.mse == pytest.approx(0.0)


def test_perfect_reconstruction_has_unit_ssim() -> None:
    x = np.random.rand(_BATCH, _IMG, _IMG, _CHANNELS).astype(np.float32)
    result = compute_reconstruction_metrics(x, x)
    assert result.ssim == pytest.approx(1.0, abs=1e-6)


def test_known_mse_value() -> None:
    # Each pixel differs by exactly 0.1 -> MSE should be 0.01
    a = np.full((1, _IMG, _IMG, _CHANNELS), 0.5, dtype=np.float32)
    b = np.full((1, _IMG, _IMG, _CHANNELS), 0.4, dtype=np.float32)
    result = compute_reconstruction_metrics(a, b)
    assert result.mse == pytest.approx(0.01, abs=1e-6)


def test_metrics_rejects_shape_mismatch() -> None:
    a = np.zeros((4, _IMG, _IMG, _CHANNELS), dtype=np.float32)
    b = np.zeros((4, _IMG, _IMG + 1, _CHANNELS), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_reconstruction_metrics(a, b)


def test_metrics_rejects_non_grayscale() -> None:
    a = np.zeros((4, _IMG, _IMG, 3), dtype=np.float32)
    b = np.zeros((4, _IMG, _IMG, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        compute_reconstruction_metrics(a, b)


def test_higher_error_produces_higher_mse() -> None:
    """Sanity: noisier reconstructions yield higher MSE."""
    rng = np.random.default_rng(0)
    x = rng.random((2, _IMG, _IMG, _CHANNELS)).astype(np.float32)
    small_noise = x + 0.05 * rng.standard_normal(x.shape).astype(np.float32)
    large_noise = x + 0.5 * rng.standard_normal(x.shape).astype(np.float32)
    small = compute_reconstruction_metrics(x, np.clip(small_noise, 0, 1))
    large = compute_reconstruction_metrics(x, np.clip(large_noise, 0, 1))
    assert large.mse > small.mse


# --- TF-dependent tests: decoder ---


def test_build_decoder_input_and_output_shapes() -> None:
    pytest.importorskip("tensorflow")
    from privacy_ml.attacks.reconstruction import build_decoder

    decoder = build_decoder(
        embedding_dim=_EMBEDDING_DIM,
        img_size=_IMG,
        channels=_CHANNELS,
        name="recon_test",
    )
    assert decoder.input_shape == (None, _EMBEDDING_DIM)
    assert decoder.output_shape == (None, _IMG, _IMG, _CHANNELS)


def test_build_decoder_forward_pass_produces_unit_range() -> None:
    pytest.importorskip("tensorflow")
    from privacy_ml.attacks.reconstruction import build_decoder

    decoder = build_decoder(
        embedding_dim=_EMBEDDING_DIM,
        img_size=_IMG,
        channels=_CHANNELS,
        name="recon_test",
    )
    z = np.random.rand(_BATCH, _EMBEDDING_DIM).astype(np.float32)
    x_hat = decoder.predict(z, verbose=0)
    assert x_hat.shape == (_BATCH, _IMG, _IMG, _CHANNELS)
    assert np.all((x_hat >= 0.0) & (x_hat <= 1.0))
