"""Shokri et al. (2017) shadow-model Membership Inference Attack.

Pipeline:
  1. Train N shadow models on disjoint (train, holdout) splits of the
     shadow pool. Each shadow produces (prediction_vector, member_flag)
     pairs from its train set (members) and holdout set (non-members).
  2. Pool all shadow pairs into a dataset for an "attack classifier"
     that maps prediction vectors to a membership probability.
  3. Apply the attack classifier to the VICTIM's prediction vectors
     on the balanced member/non-member eval set.

For the pneumonia binary task, the victim's prediction vector is a
scalar sigmoid probability. We turn it into a 2-d feature
[p, 1-p] so the attack classifier can learn asymmetric thresholds on
either side (members with both very-high and very-low confidence).

The attack classifier is a small MLP trained with Keras/TF, since
that's the rest of the stack's framework. If TF is unavailable
(unit tests running without GPU deps), callers can skip the training
step and provide a pre-trained model.

Reference: Shokri, R., Stronati, M., Song, C., & Shmatikov, V. (2017).
"Membership Inference Attacks Against Machine Learning Models."
2017 IEEE Symposium on Security and Privacy (SP).
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from privacy_ml.metrics import auc_score


class AttackTrainingData(NamedTuple):
    """Pooled (feature, member) pairs from all shadow models.

    features: shape (N, 2), float32
    member_labels: shape (N,), int {0, 1}
    """

    features: np.ndarray
    member_labels: np.ndarray


class ShokriAttackResult(NamedTuple):
    """Outcome of the Shokri attack on the victim eval set."""

    attack_accuracy: float
    attack_auc: float


def build_features_from_probabilities(probabilities: np.ndarray) -> np.ndarray:
    """Expand scalar sigmoid probs into 2-d features [p, 1-p].

    Gives the attack classifier a symmetric view of the victim's output
    — it can learn that both very-high and very-low confidence on the
    TRUE class are member-discriminating signals.
    """
    p = probabilities.reshape(-1, 1).astype(np.float32)
    return np.concatenate([p, 1.0 - p], axis=1)


def assemble_attack_training_data(
    shadow_member_probs: np.ndarray,
    shadow_nonmember_probs: np.ndarray,
) -> AttackTrainingData:
    """Pool all shadows' (prediction, member_label) pairs.

    Parameters
    ----------
    shadow_member_probs
        Concatenation across all shadow models of their *train* set
        prediction probabilities (members). Shape (M,).
    shadow_nonmember_probs
        Concatenation across all shadow models of their *holdout* set
        prediction probabilities (non-members). Shape (H,).

    Returns
    -------
    AttackTrainingData with features (M+H, 2) and member_labels (M+H,).
    """
    features = np.concatenate(
        [
            build_features_from_probabilities(shadow_member_probs),
            build_features_from_probabilities(shadow_nonmember_probs),
        ],
        axis=0,
    )
    labels = np.concatenate(
        [
            np.ones(len(shadow_member_probs), dtype=np.int64),
            np.zeros(len(shadow_nonmember_probs), dtype=np.int64),
        ],
        axis=0,
    )
    return AttackTrainingData(features=features, member_labels=labels)


def build_attack_classifier():
    """Small MLP: 2 → 64 → 32 → 1 (sigmoid). Keras import kept local."""
    from tensorflow.keras import Input, Model, layers

    inputs = Input(shape=(2,), name="attack_input")
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs, name="shokri_attack")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def evaluate_attack(
    attack_model_predict_fn,
    victim_probs: np.ndarray,
    member_mask: np.ndarray,
) -> ShokriAttackResult:
    """Apply a trained attack classifier to the victim's eval predictions.

    Parameters
    ----------
    attack_model_predict_fn
        Callable taking (N, 2) features, returning (N,) or (N, 1) member
        probabilities. Injection allows unit-testing without TF by
        supplying a stub.
    victim_probs
        Victim's sigmoid outputs on the balanced eval set. Shape (N,).
    member_mask
        Ground truth membership for eval queries. Shape (N,), bool.
    """
    features = build_features_from_probabilities(victim_probs)
    attack_scores = np.asarray(attack_model_predict_fn(features)).reshape(-1)
    predicted_member = attack_scores >= 0.5
    member_bool = member_mask.astype(bool)
    accuracy = float(np.mean(predicted_member == member_bool))
    auc = auc_score(attack_scores, member_bool.astype(int))
    return ShokriAttackResult(
        attack_accuracy=accuracy,
        attack_auc=float(auc),
    )
