"""Utility, privacy, and efficiency metrics per spec §8.

Utility:
  - test_accuracy  : fraction correct at threshold 0.5
  - f1             : F1 on PNEUMONIA (positive class)
  - ece            : Expected Calibration Error

Privacy metrics are computed by the attacks themselves (see
privacy_ml.attacks); this module only provides helpers for packaging
their outputs into the run-record schema.

Efficiency:
  - latency/memory are recorded by the runner, not computed here.
  - embedding_bytes_per_query is a constant = EMBEDDING_DIM × 4 (float32).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

_PREDICTION_THRESHOLD: float = 0.5
_DEFAULT_ECE_BINS: int = 10


class UtilityMetrics(NamedTuple):
    """Utility-side metrics for one PPT configuration."""

    test_accuracy: float
    f1: float
    ece: float


def accuracy(y_true: np.ndarray, y_pred_prob: np.ndarray) -> float:
    """Binary accuracy at the canonical 0.5 threshold."""
    y_pred = (y_pred_prob >= _PREDICTION_THRESHOLD).astype(np.int64)
    return float(np.mean(y_pred == y_true.astype(np.int64)))


def f1_score_binary(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    positive_label: int,
) -> float:
    """F1 for a binary classifier, treating `positive_label` as positive.

    F1 = 2 * precision * recall / (precision + recall). Returns 0.0
    when no positives are predicted AND no positives exist (degenerate).
    """
    y_pred = (y_pred_prob >= _PREDICTION_THRESHOLD).astype(np.int64)
    y_true_int = y_true.astype(np.int64)
    tp = float(np.sum((y_pred == positive_label) & (y_true_int == positive_label)))
    fp = float(np.sum((y_pred == positive_label) & (y_true_int != positive_label)))
    fn = float(np.sum((y_pred != positive_label) & (y_true_int == positive_label)))
    denom = 2.0 * tp + fp + fn
    if denom == 0.0:
        return 0.0
    return (2.0 * tp) / denom


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    n_bins: int,
) -> float:
    """ECE: weighted average over confidence bins of |accuracy - confidence|.

    Low ECE ⇒ the model's confidence is well-calibrated against its
    actual accuracy. High ECE under privacy noise indicates overconfident
    wrong predictions (spec §8 target: track this, no hard threshold).
    """
    y_true_int = y_true.astype(np.int64)
    confidence = np.where(
        y_pred_prob >= _PREDICTION_THRESHOLD,
        y_pred_prob,
        1.0 - y_pred_prob,
    )
    correct = (
        (y_pred_prob >= _PREDICTION_THRESHOLD).astype(np.int64) == y_true_int
    ).astype(np.float64)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    total_weighted_gap = 0.0
    n_total = len(y_true)
    for bin_idx in range(n_bins):
        lower = bin_edges[bin_idx]
        upper = bin_edges[bin_idx + 1]
        if bin_idx == n_bins - 1:
            in_bin = (confidence >= lower) & (confidence <= upper)
        else:
            in_bin = (confidence >= lower) & (confidence < upper)
        bin_count = int(np.sum(in_bin))
        if bin_count == 0:
            continue
        bin_confidence = float(np.mean(confidence[in_bin]))
        bin_accuracy = float(np.mean(correct[in_bin]))
        weight = bin_count / n_total
        total_weighted_gap += weight * abs(bin_accuracy - bin_confidence)
    return float(total_weighted_gap)


def compute_utility_metrics(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    positive_label: int,
    ece_bins: int,
) -> UtilityMetrics:
    """Bundle the three utility metrics for one PPT configuration."""
    return UtilityMetrics(
        test_accuracy=accuracy(y_true, y_pred_prob),
        f1=f1_score_binary(y_true, y_pred_prob, positive_label),
        ece=expected_calibration_error(y_true, y_pred_prob, ece_bins),
    )
