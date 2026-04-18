"""Tests for utility metrics: accuracy, F1, ECE.

ECE has a well-known edge case where a perfectly-calibrated prediction
produces ece=0 only if confidence bins are aligned with the data; this
suite covers the pathological cases and the common case.
"""
from __future__ import annotations

import numpy as np
import pytest

from privacy_ml.data import PNEUMONIA_LABEL
from privacy_ml.metrics import (
    UtilityMetrics,
    accuracy,
    auc_score,
    compute_utility_metrics,
    expected_calibration_error,
    f1_score_binary,
)

_ECE_BINS: int = 10


def test_accuracy_perfect_predictions() -> None:
    y_true = np.array([1, 0, 1, 0, 1])
    y_pred = np.array([0.9, 0.1, 0.8, 0.2, 0.95])
    assert accuracy(y_true, y_pred) == 1.0


def test_accuracy_all_wrong() -> None:
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8])
    assert accuracy(y_true, y_pred) == 0.0


def test_accuracy_threshold_at_half() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.5, 0.5, 0.5, 0.5])
    # 0.5 predicts 1 (>= threshold); matches 2 of 4
    assert accuracy(y_true, y_pred) == 0.5


def test_f1_perfect_recall_and_precision() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.9, 0.9, 0.1, 0.1])
    assert f1_score_binary(y_true, y_pred, positive_label=PNEUMONIA_LABEL) == 1.0


def test_f1_is_zero_when_no_positives_predicted_or_present() -> None:
    y_true = np.array([0, 0, 0, 0])
    y_pred = np.array([0.1, 0.2, 0.3, 0.4])
    assert (
        f1_score_binary(y_true, y_pred, positive_label=PNEUMONIA_LABEL) == 0.0
    )


def test_f1_typical_imbalanced_case() -> None:
    # TP=2, FP=1, FN=1 → precision=2/3, recall=2/3 → f1=2/3
    y_true = np.array([1, 1, 1, 0])
    y_pred = np.array([0.9, 0.9, 0.1, 0.8])
    f1 = f1_score_binary(y_true, y_pred, positive_label=PNEUMONIA_LABEL)
    assert f1 == pytest.approx(2.0 / 3.0, abs=1e-6)


def test_ece_zero_on_perfect_calibration_and_prediction() -> None:
    # Confidence 1.0, all correct ⇒ ECE = 0
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    y_pred = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    ece = expected_calibration_error(y_true, y_pred, n_bins=_ECE_BINS)
    assert ece == pytest.approx(0.0, abs=1e-6)


def test_ece_high_for_overconfident_wrong() -> None:
    # All predictions are 0.95 confident but half are wrong ⇒ large ECE
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.95, 0.95, 0.95, 0.95])
    ece = expected_calibration_error(y_true, y_pred, n_bins=_ECE_BINS)
    assert ece >= 0.4


def test_ece_is_in_unit_interval() -> None:
    rng = np.random.default_rng(0)
    y_true = rng.integers(0, 2, size=100)
    y_pred = rng.uniform(0.0, 1.0, size=100)
    ece = expected_calibration_error(y_true, y_pred, n_bins=_ECE_BINS)
    assert 0.0 <= ece <= 1.0


def test_auc_returns_half_on_single_class() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([1, 1, 1, 1])
    assert auc_score(scores, labels) == 0.5


def test_auc_is_one_on_perfect_separation() -> None:
    scores = np.array([0.1, 0.2, 0.9, 0.95])
    labels = np.array([0, 0, 1, 1])
    assert auc_score(scores, labels) == pytest.approx(1.0)


def test_auc_is_zero_on_reversed_separation() -> None:
    scores = np.array([0.9, 0.95, 0.1, 0.2])
    labels = np.array([0, 0, 1, 1])
    assert auc_score(scores, labels) == pytest.approx(0.0)


def test_auc_handles_tied_scores() -> None:
    # Two tied scores at 0.5, one class per side of the tie.
    # sklearn averages ranks; expected AUC = 0.5 (no separation signal).
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    labels = np.array([0, 0, 1, 1])
    assert auc_score(scores, labels) == pytest.approx(0.5)


def test_compute_utility_metrics_bundles_three_values() -> None:
    y_true = np.array([1, 1, 0, 0])
    y_pred = np.array([0.9, 0.9, 0.1, 0.1])
    metrics = compute_utility_metrics(
        y_true, y_pred, positive_label=PNEUMONIA_LABEL, ece_bins=_ECE_BINS
    )
    assert isinstance(metrics, UtilityMetrics)
    assert metrics.test_accuracy == 1.0
    assert metrics.f1 == 1.0
    assert 0.0 <= metrics.ece <= 1.0
