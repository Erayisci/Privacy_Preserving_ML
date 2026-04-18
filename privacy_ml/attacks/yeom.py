"""Yeom et al. (2018) threshold-based Membership Inference Attack.

Idea: a victim model typically assigns lower loss to samples in its
training set than to unseen samples. The attacker picks a threshold on
loss (or equivalently, on per-sample confidence) and predicts
"member" iff loss < threshold.

No shadow models required; this is the cheapest MIA variant. Zero
extra training time — the attacker only queries the already-trained
victim.

Reference: Yeom, S., Giacomelli, I., Fredrikson, M., & Jha, S. (2018).
"Privacy Risk in Machine Learning: Analyzing the Connection to
Overfitting." 2018 IEEE 31st Computer Security Foundations Symposium.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

_BCE_EPSILON: float = 1e-7


class YeomAttackResult(NamedTuple):
    """Outcome of running the Yeom threshold attack on an eval set."""

    attack_accuracy: float
    attack_auc: float
    threshold: float


def binary_cross_entropy(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
) -> np.ndarray:
    """Per-sample binary cross-entropy loss (used as the attack signal).

    y_pred_prob is clipped to avoid log(0).
    Returns an array of shape (N,) with one loss per sample.
    """
    clipped = np.clip(y_pred_prob, _BCE_EPSILON, 1.0 - _BCE_EPSILON)
    return -(
        y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped)
    )


def _auc_from_scores(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC-AUC for binary labels by score. Higher score ⇒ more likely class 1.

    Uses the rank formula (Mann-Whitney U) so no sklearn dependency here.
    """
    if len(np.unique(labels)) < 2:
        return 0.5
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = float(np.sum(labels == 1))
    n_neg = float(np.sum(labels == 0))
    sum_ranks_pos = float(np.sum(ranks[labels == 1]))
    u = sum_ranks_pos - n_pos * (n_pos + 1.0) / 2.0
    return u / (n_pos * n_neg)


def attack(
    y_true: np.ndarray,
    y_pred_prob: np.ndarray,
    member_mask: np.ndarray,
) -> YeomAttackResult:
    """Run the Yeom threshold attack on a balanced member/non-member set.

    Parameters
    ----------
    y_true
        Ground-truth binary labels on the evaluation queries (shape (N,)).
    y_pred_prob
        Victim model's sigmoid output for the same queries (shape (N,),
        values in [0, 1]).
    member_mask
        Boolean array (shape (N,)) where True = sample was in the
        victim's training set, False = held-out.

    Returns
    -------
    YeomAttackResult with attack_accuracy, attack_auc, and the
    loss-threshold that maximised accuracy on this eval set.
    """
    losses = binary_cross_entropy(y_true.astype(np.float64), y_pred_prob.astype(np.float64))

    member_mask_bool = member_mask.astype(bool)

    # Scan candidate thresholds = all observed loss values (+ ±ε).
    # Members should have LOWER loss ⇒ "loss <= threshold" predicts member.
    candidate_thresholds = np.sort(np.unique(losses))
    best_accuracy = 0.0
    best_threshold = float(candidate_thresholds[0])
    for tau in candidate_thresholds:
        predicted_member = losses <= tau
        accuracy = float(np.mean(predicted_member == member_mask_bool))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = float(tau)

    # AUC uses the continuous "member score" = -loss (higher = more member-like).
    member_score = -losses
    auc = _auc_from_scores(member_score, member_mask_bool.astype(int))

    return YeomAttackResult(
        attack_accuracy=best_accuracy,
        attack_auc=float(auc),
        threshold=best_threshold,
    )
