"""Tests for Yeom threshold attack + Shokri shadow-model attack.

Sanity invariants:
  - On a perfectly overfit synthetic victim, Yeom attack accuracy ≥ 90%.
  - On a well-generalized victim (identical train/holdout distributions),
    Yeom attack accuracy ≈ 50% (random guessing).
  - Binary cross-entropy matches sklearn's log_loss within 1e-6.
  - Shokri's feature builder and dataset assembly preserve shapes.
  - AUC is in [0, 1] and returns 0.5 on a single-class input.
"""
from __future__ import annotations

import numpy as np
import pytest

from privacy_ml.attacks.shokri import (
    AttackTrainingData,
    ShokriAttackResult,
    assemble_attack_training_data,
    build_features_from_probabilities,
    evaluate_attack,
)
from privacy_ml.attacks.yeom import (
    YeomAttackResult,
    _auc_from_scores,
    attack as yeom_attack,
    binary_cross_entropy,
)

_EVAL_SIZE_PER_HALF: int = 200
_BCE_TOLERANCE: float = 1e-6


def _build_overfit_scenario(seed: int) -> tuple:
    """Members are near-perfectly predicted; non-members are noisy.

    This is the classic MIA-vulnerable case: high train accuracy,
    lower held-out accuracy.
    """
    rng = np.random.default_rng(seed)
    n = _EVAL_SIZE_PER_HALF
    y_true_members = rng.integers(0, 2, size=n).astype(np.int64)
    y_true_nonmembers = rng.integers(0, 2, size=n).astype(np.int64)

    # Members: prediction is confident and correct (low loss)
    prob_members = np.where(
        y_true_members == 1,
        rng.uniform(0.9, 0.99, size=n),
        rng.uniform(0.01, 0.1, size=n),
    )
    # Non-members: prediction is noisier / less confident (higher loss on average)
    prob_nonmembers = rng.uniform(0.3, 0.7, size=n)

    y_true = np.concatenate([y_true_members, y_true_nonmembers])
    y_pred = np.concatenate([prob_members, prob_nonmembers])
    member_mask = np.concatenate([np.ones(n, dtype=bool), np.zeros(n, dtype=bool)])
    return y_true, y_pred, member_mask


def _build_generalized_scenario(seed: int) -> tuple:
    """Members and non-members look identical — a well-generalized victim."""
    rng = np.random.default_rng(seed)
    n = _EVAL_SIZE_PER_HALF
    y_true = rng.integers(0, 2, size=2 * n).astype(np.int64)
    y_pred = np.where(
        y_true == 1,
        rng.uniform(0.6, 0.9, size=2 * n),
        rng.uniform(0.1, 0.4, size=2 * n),
    )
    member_mask = np.concatenate([np.ones(n, dtype=bool), np.zeros(n, dtype=bool)])
    return y_true, y_pred, member_mask


# --- Yeom attack ---


def test_bce_matches_manual_computation() -> None:
    y_true = np.array([1, 0, 1, 0], dtype=np.float64)
    y_pred = np.array([0.9, 0.2, 0.4, 0.8], dtype=np.float64)
    bce = binary_cross_entropy(y_true, y_pred)
    expected = -np.array(
        [
            np.log(0.9),
            np.log(1 - 0.2),
            np.log(0.4),
            np.log(1 - 0.8),
        ]
    )
    assert np.allclose(bce, expected, atol=_BCE_TOLERANCE)


def test_bce_clips_extreme_probabilities() -> None:
    y_true = np.array([1, 0], dtype=np.float64)
    y_pred = np.array([0.0, 1.0], dtype=np.float64)
    bce = binary_cross_entropy(y_true, y_pred)
    assert np.all(np.isfinite(bce))


def test_yeom_attacks_overfit_victim_succeeds() -> None:
    y_true, y_pred, member_mask = _build_overfit_scenario(seed=0)
    result = yeom_attack(y_true, y_pred, member_mask)
    assert isinstance(result, YeomAttackResult)
    assert result.attack_accuracy >= 0.90


def test_yeom_on_generalized_victim_is_near_chance() -> None:
    y_true, y_pred, member_mask = _build_generalized_scenario(seed=0)
    result = yeom_attack(y_true, y_pred, member_mask)
    assert 0.40 <= result.attack_accuracy <= 0.60


def test_yeom_auc_bounds() -> None:
    y_true, y_pred, member_mask = _build_overfit_scenario(seed=1)
    result = yeom_attack(y_true, y_pred, member_mask)
    assert 0.0 <= result.attack_auc <= 1.0


def test_yeom_overfit_auc_is_high() -> None:
    y_true, y_pred, member_mask = _build_overfit_scenario(seed=2)
    result = yeom_attack(y_true, y_pred, member_mask)
    assert result.attack_auc >= 0.90


# --- AUC helper ---


def test_auc_returns_half_on_single_class() -> None:
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    labels = np.array([1, 1, 1, 1])
    assert _auc_from_scores(scores, labels) == 0.5


def test_auc_is_one_on_perfect_separation() -> None:
    scores = np.array([0.1, 0.2, 0.9, 0.95])
    labels = np.array([0, 0, 1, 1])
    auc = _auc_from_scores(scores, labels)
    assert auc == pytest.approx(1.0)


def test_auc_is_zero_on_reversed_separation() -> None:
    scores = np.array([0.9, 0.95, 0.1, 0.2])
    labels = np.array([0, 0, 1, 1])
    auc = _auc_from_scores(scores, labels)
    assert auc == pytest.approx(0.0)


# --- Shokri attack (framework-free bits) ---


def test_build_features_creates_p_and_one_minus_p() -> None:
    probs = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    features = build_features_from_probabilities(probs)
    assert features.shape == (3, 2)
    np.testing.assert_allclose(features[:, 0] + features[:, 1], 1.0, atol=1e-6)


def test_assemble_attack_training_data_shape_and_labels() -> None:
    member_probs = np.array([0.9, 0.85, 0.95], dtype=np.float32)
    nonmember_probs = np.array([0.6, 0.4], dtype=np.float32)
    data = assemble_attack_training_data(member_probs, nonmember_probs)
    assert isinstance(data, AttackTrainingData)
    assert data.features.shape == (5, 2)
    assert data.member_labels.shape == (5,)
    assert np.array_equal(
        data.member_labels, np.array([1, 1, 1, 0, 0], dtype=np.int64)
    )


def test_evaluate_attack_with_stub_predict_fn() -> None:
    # Deterministic stub: attack says "member" iff p >= 0.7
    def stub_predict(features: np.ndarray) -> np.ndarray:
        return (features[:, 0] >= 0.7).astype(np.float32)

    victim_probs = np.array([0.9, 0.3, 0.8, 0.1], dtype=np.float32)
    member_mask = np.array([True, False, True, False])
    result = evaluate_attack(stub_predict, victim_probs, member_mask)
    assert isinstance(result, ShokriAttackResult)
    assert result.attack_accuracy == 1.0
    assert 0.0 <= result.attack_auc <= 1.0
