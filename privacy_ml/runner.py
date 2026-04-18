"""Pipeline runner — the orchestration layer.

Ties everything together:
  1. Load data + produce canonical splits
  2. Train or load victim encoder+head (cached by encoder_hash)
  3. Produce embeddings for victim splits (applying image-layer PPTs first)
  4. Apply embedding-layer PPTs (DP, SMPC); retrain head if any active
  5. Predict on a balanced MIA evaluation set
  6. Compute utility metrics on victim_nonmembers
  7. Run Yeom + Shokri attacks (Shokri trains 5 shadow models)
  8. Package latency/memory/bandwidth efficiency metrics
  9. Return RunResult (caller persists to results/runs.jsonl)

TensorFlow is imported lazily so that non-TF consumers (e.g., offline
tests of RunConfig / CLI parsing) don't pay the import cost.

See docs/superpowers/specs/2026-04-18-mia-design.md §5–§8.
"""
from __future__ import annotations

import json
import resource
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, NamedTuple, Optional, Tuple

import numpy as np

from .attacks.shokri import (
    ShokriAttackResult,
    assemble_attack_training_data,
    evaluate_attack,
)
from .attacks.yeom import YeomAttackResult, attack as run_yeom_attack
from .cache import (
    CachePaths,
    cache_paths,
    encoder_hash,
    is_encoder_cached,
    read_encoder_meta,
    write_encoder_meta,
)
from .data import (
    PNEUMONIA_LABEL,
    ShadowSplit,
    SplitIndices,
    build_shadow_splits,
    load_kaggle_origins,
    load_kaggle_pool,
    split_pool_indices,
)
from .metrics import compute_utility_metrics
from .models import (
    DEFAULT_CHANNELS,
    DEFAULT_DROPOUT_RATE,
    DEFAULT_IMG_SIZE,
    DEFAULT_LEARNING_RATE,
    EMBEDDING_DIM,
)
from .ppt.base import PrivacyMechanism

BATCH_SIZE: int = 32
ECE_BINS: int = 10
N_SHADOWS: int = 5
SHADOW_TRAIN_SIZE: int = 500
SHADOW_HOLDOUT_SIZE: int = 500
MIA_EVAL_PER_HALF: int = 500
ATTACK_CLASSIFIER_EPOCHS: int = 20
SHADOW_TRAINING_SEED_OFFSET: int = 100
EMBEDDING_BYTES_PER_QUERY: int = EMBEDDING_DIM * 4  # 128 * 4 bytes


@dataclass(frozen=True)
class RunConfig:
    """Complete configuration for one pipeline invocation (from CLI flags)."""

    # PPT flags
    dp_enabled: bool
    dp_epsilon: float
    bie_enabled: bool
    bie_key_seed: int
    bie_tile_size: int
    smpc_enabled: bool
    smpc_shares: int
    # MIA variants
    run_yeom: bool
    run_shokri: bool
    # Training
    epochs: int
    seed: int
    # Paths
    data_dir: Path
    cache_dir: Path
    output_dir: Path
    # Metadata
    tag: str

    @property
    def _hash_tile_size(self) -> int:
        """Feed 0 when BIE is off (matches spec §5 convention)."""
        return self.bie_tile_size if self.bie_enabled else 0

    @property
    def victim_encoder_hash(self) -> str:
        return encoder_hash(
            bie_on=self.bie_enabled,
            bie_key_seed=self.bie_key_seed,
            bie_tile_size=self._hash_tile_size,
            training_seed=self.seed,
            epochs=self.epochs,
        )

    def shadow_encoder_hash(self, shadow_idx: int) -> str:
        return encoder_hash(
            bie_on=self.bie_enabled,
            bie_key_seed=self.bie_key_seed,
            bie_tile_size=self._hash_tile_size,
            training_seed=self.seed + SHADOW_TRAINING_SEED_OFFSET + shadow_idx,
            epochs=self.epochs,
        )


class UtilityResult(NamedTuple):
    test_accuracy: float
    f1: float
    ece: float


class PrivacyResult(NamedTuple):
    yeom: Optional[YeomAttackResult]
    shokri: Optional[ShokriAttackResult]


class EfficiencyResult(NamedTuple):
    train_latency_seconds: float
    inference_latency_ms_per_query: float
    memory_peak_mb: float
    embedding_bytes_per_query: int


@dataclass(frozen=True)
class RunResult:
    config: RunConfig
    utility: UtilityResult
    privacy: PrivacyResult
    efficiency: EfficiencyResult
    encoder_hash_id: str
    timestamp: str


# --- PPT application helpers ---


def _fit_image_ppts(
    training_images: np.ndarray,
    image_ppts: List[PrivacyMechanism],
) -> None:
    """Fit image-layer PPTs on training images (usually a no-op for BIE)."""
    for ppt in image_ppts:
        if ppt.layer != "image":
            raise ValueError(
                f"Expected image-layer PPT, got layer={ppt.layer}"
            )
        ppt.fit(training_images)


def _apply_image_ppts(
    images: np.ndarray,
    image_ppts: List[PrivacyMechanism],
) -> np.ndarray:
    """Run image-layer PPTs in order. Must be pre-fit."""
    out = images
    for ppt in image_ppts:
        out = ppt.apply(out)
    return out


def _fit_embedding_ppts(
    training_embeddings: np.ndarray,
    embedding_ppts: List[PrivacyMechanism],
) -> None:
    """Fit embedding-layer PPTs on clean training embeddings."""
    for ppt in embedding_ppts:
        if ppt.layer != "embedding":
            raise ValueError(
                f"Expected embedding-layer PPT, got layer={ppt.layer}"
            )
        ppt.fit(training_embeddings)


def _apply_embedding_ppts(
    embeddings: np.ndarray,
    embedding_ppts: List[PrivacyMechanism],
) -> np.ndarray:
    """Run embedding-layer PPTs in order. Must be pre-fit."""
    out = embeddings
    for ppt in embedding_ppts:
        out = ppt.apply(out)
    return out


# --- Encoder + head training / caching ---


def _head_weights_path(encoder_paths: CachePaths) -> Path:
    """Sibling path to the encoder weights, storing the jointly-trained head."""
    return encoder_paths.encoder_weights.with_suffix(".head.keras")


def _train_encoder_and_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    seed: int,
):
    """Joint encoder+head training; matches notebook cell 4 setup."""
    import tensorflow as tf

    from .models import (
        build_encoder,
        build_end_to_end,
        build_head,
        compile_for_binary_classification,
    )

    tf.keras.utils.set_random_seed(seed)
    encoder = build_encoder(
        DEFAULT_IMG_SIZE, DEFAULT_CHANNELS, EMBEDDING_DIM, "encoder"
    )
    head = build_head(EMBEDDING_DIM, DEFAULT_DROPOUT_RATE, "head")
    pipeline = build_end_to_end(encoder, head, "pipeline")
    compile_for_binary_classification(pipeline, DEFAULT_LEARNING_RATE)
    pipeline.fit(
        X_train,
        y_train.astype(np.float32),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    return encoder, head


def _save_encoder_and_head(
    encoder,
    head,
    paths: CachePaths,
    config_meta: dict,
) -> None:
    """Persist both models + meta.json to the cache directory."""
    paths.encoder_weights.parent.mkdir(parents=True, exist_ok=True)
    encoder.save(paths.encoder_weights)
    head.save(_head_weights_path(paths))
    write_encoder_meta(paths.encoder_meta, config_meta)


def _load_encoder_and_head(paths: CachePaths):
    import tensorflow as tf

    encoder = tf.keras.models.load_model(paths.encoder_weights)
    head = tf.keras.models.load_model(_head_weights_path(paths))
    return encoder, head


def _has_cached_encoder_and_head(paths: CachePaths) -> bool:
    return is_encoder_cached(paths) and _head_weights_path(paths).exists()


def _load_or_train_encoder_and_head(
    X: np.ndarray,
    y: np.ndarray,
    train_indices: np.ndarray,
    image_ppts: List[PrivacyMechanism],
    epochs: int,
    training_seed: int,
    paths: CachePaths,
    config_meta: dict,
) -> Tuple:
    """Cache-aware encoder+head training entry point.

    On cache hit: load from disk (fast).
    On cache miss: apply image-layer PPTs to training images, train
    encoder+head jointly, save to cache.
    """
    if _has_cached_encoder_and_head(paths):
        return _load_encoder_and_head(paths)

    X_train = X[train_indices]
    y_train = y[train_indices]
    if image_ppts:
        _fit_image_ppts(X_train, image_ppts)
        X_train = _apply_image_ppts(X_train, image_ppts)

    encoder, head = _train_encoder_and_head(
        X_train, y_train, epochs, training_seed
    )
    _save_encoder_and_head(encoder, head, paths, config_meta)
    return encoder, head


# --- Embedding extraction ---


def _encode_indices(
    encoder,
    X: np.ndarray,
    indices: np.ndarray,
    image_ppts: List[PrivacyMechanism],
) -> np.ndarray:
    """Produce embeddings for `indices` into the raw image pool.

    image_ppts (BIE) are applied before encoding so the embeddings
    reflect the full image-layer privacy pipeline.
    """
    batch = X[indices]
    if image_ppts:
        batch = _apply_image_ppts(batch, image_ppts)
    return np.asarray(encoder.predict(batch, verbose=0), dtype=np.float32)


# --- Head retraining for embedding-layer PPTs ---


def _train_head_on_embeddings(
    E_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int,
    seed: int,
):
    """Fresh head trained on (possibly-privatized) embeddings.

    Used when DP/SMPC are active — the baseline head was trained on
    clean embeddings and wouldn't make calibrated predictions on noisy
    ones.
    """
    import tensorflow as tf

    from .models import build_head, compile_for_binary_classification

    tf.keras.utils.set_random_seed(seed)
    head = build_head(EMBEDDING_DIM, DEFAULT_DROPOUT_RATE, "head_retrained")
    compile_for_binary_classification(head, DEFAULT_LEARNING_RATE)
    head.fit(
        E_train,
        y_train.astype(np.float32),
        epochs=epochs,
        batch_size=BATCH_SIZE,
        verbose=0,
    )
    return head


# --- Shokri shadow-model MIA ---


def _build_shokri_attack_classifier():
    from tensorflow.keras import Input, Model, layers

    inputs = Input(shape=(2,), name="attack_input")
    x = layers.Dense(64, activation="relu")(inputs)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inputs, outputs=outputs, name="shokri_attack")
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model


def _run_shokri_attack(
    X: np.ndarray,
    y: np.ndarray,
    shadow_pool_indices: np.ndarray,
    image_ppts: List[PrivacyMechanism],
    embedding_ppts: List[PrivacyMechanism],
    victim_eval_probs: np.ndarray,
    eval_member_mask: np.ndarray,
    config: RunConfig,
) -> ShokriAttackResult:
    """Train 5 shadows, pool their (pred, member) pairs, train attack MLP, evaluate on victim."""
    shadow_splits = build_shadow_splits(
        shadow_indices=shadow_pool_indices,
        y=y,
        n_shadows=N_SHADOWS,
        train_size=SHADOW_TRAIN_SIZE,
        holdout_size=SHADOW_HOLDOUT_SIZE,
        seed=config.seed,
    )

    all_member_probs: List[np.ndarray] = []
    all_nonmember_probs: List[np.ndarray] = []

    for shadow_idx, shadow_split in enumerate(shadow_splits):
        shadow_paths = cache_paths(
            config.cache_dir, config.shadow_encoder_hash(shadow_idx)
        )
        shadow_training_seed = (
            config.seed + SHADOW_TRAINING_SEED_OFFSET + shadow_idx
        )
        shadow_meta = {
            "role": "shadow",
            "shadow_idx": shadow_idx,
            "victim_tag": config.tag,
        }
        encoder, baseline_head = _load_or_train_encoder_and_head(
            X=X,
            y=y,
            train_indices=shadow_split.train_indices,
            image_ppts=image_ppts,
            epochs=config.epochs,
            training_seed=shadow_training_seed,
            paths=shadow_paths,
            config_meta=shadow_meta,
        )

        E_train = _encode_indices(
            encoder, X, shadow_split.train_indices, image_ppts
        )
        E_holdout = _encode_indices(
            encoder, X, shadow_split.holdout_indices, image_ppts
        )

        if embedding_ppts:
            _fit_embedding_ppts(E_train, embedding_ppts)
            E_train_priv = _apply_embedding_ppts(E_train, embedding_ppts)
            E_holdout_priv = _apply_embedding_ppts(E_holdout, embedding_ppts)
            head = _train_head_on_embeddings(
                E_train_priv,
                y[shadow_split.train_indices],
                config.epochs,
                shadow_training_seed,
            )
        else:
            E_train_priv = E_train
            E_holdout_priv = E_holdout
            head = baseline_head

        probs_train = np.asarray(
            head.predict(E_train_priv, verbose=0), dtype=np.float32
        ).reshape(-1)
        probs_holdout = np.asarray(
            head.predict(E_holdout_priv, verbose=0), dtype=np.float32
        ).reshape(-1)
        all_member_probs.append(probs_train)
        all_nonmember_probs.append(probs_holdout)

    shadow_members = np.concatenate(all_member_probs)
    shadow_nonmembers = np.concatenate(all_nonmember_probs)
    attack_data = assemble_attack_training_data(
        shadow_members, shadow_nonmembers
    )

    attack_model = _build_shokri_attack_classifier()
    attack_model.fit(
        attack_data.features,
        attack_data.member_labels.astype(np.float32),
        epochs=ATTACK_CLASSIFIER_EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=0,
    )

    return evaluate_attack(
        lambda feats: attack_model.predict(feats, verbose=0),
        victim_eval_probs,
        eval_member_mask,
    )


# --- Main orchestrator ---


def _sample_indices(
    total: int, n_samples: int, rng: np.random.Generator
) -> np.ndarray:
    if n_samples >= total:
        return np.arange(total)
    return rng.choice(total, n_samples, replace=False)


def _peak_memory_mb() -> float:
    """Cross-platform peak RSS. Returns 0.0 if unavailable.

    getrusage reports maxrss in bytes on macOS and kilobytes on Linux.
    """
    try:
        maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (OSError, AttributeError):
        return 0.0
    if sys.platform == "darwin":
        return float(maxrss) / (1024.0 * 1024.0)
    return float(maxrss) / 1024.0


def run_single_config(
    config: RunConfig,
    image_ppts: List[PrivacyMechanism],
    embedding_ppts: List[PrivacyMechanism],
) -> RunResult:
    """Execute one pipeline configuration end-to-end. Returns a RunResult."""
    X, y = load_kaggle_pool(config.data_dir, DEFAULT_IMG_SIZE)
    subdirs = load_kaggle_origins(config.data_dir)
    splits = split_pool_indices(y, subdirs, seed=config.seed)

    victim_paths = cache_paths(config.cache_dir, config.victim_encoder_hash)
    victim_meta = {
        "role": "victim",
        "tag": config.tag,
        "config": {
            "bie_enabled": config.bie_enabled,
            "bie_key_seed": config.bie_key_seed,
            "seed": config.seed,
            "epochs": config.epochs,
        },
    }

    train_start = time.perf_counter()
    encoder, baseline_head = _load_or_train_encoder_and_head(
        X=X,
        y=y,
        train_indices=splits.victim_members,
        image_ppts=image_ppts,
        epochs=config.epochs,
        training_seed=config.seed,
        paths=victim_paths,
        config_meta=victim_meta,
    )
    train_latency = time.perf_counter() - train_start

    E_members = _encode_indices(encoder, X, splits.victim_members, image_ppts)
    E_nonmem = _encode_indices(encoder, X, splits.victim_nonmembers, image_ppts)

    if embedding_ppts:
        _fit_embedding_ppts(E_members, embedding_ppts)
        E_members_priv = _apply_embedding_ppts(E_members, embedding_ppts)
        E_nonmem_priv = _apply_embedding_ppts(E_nonmem, embedding_ppts)
        head = _train_head_on_embeddings(
            E_members_priv,
            y[splits.victim_members],
            config.epochs,
            config.seed,
        )
    else:
        E_members_priv = E_members
        E_nonmem_priv = E_nonmem
        head = baseline_head

    inference_start = time.perf_counter()
    probs_members = np.asarray(
        head.predict(E_members_priv, verbose=0), dtype=np.float32
    ).reshape(-1)
    probs_nonmem = np.asarray(
        head.predict(E_nonmem_priv, verbose=0), dtype=np.float32
    ).reshape(-1)
    n_inference_queries = len(probs_members) + len(probs_nonmem)
    inference_latency_ms = (
        (time.perf_counter() - inference_start) / n_inference_queries * 1000.0
    )

    utility_metrics = compute_utility_metrics(
        y[splits.victim_nonmembers],
        probs_nonmem,
        positive_label=PNEUMONIA_LABEL,
        ece_bins=ECE_BINS,
    )
    utility = UtilityResult(
        test_accuracy=utility_metrics.test_accuracy,
        f1=utility_metrics.f1,
        ece=utility_metrics.ece,
    )

    eval_rng = np.random.default_rng(config.seed + 1000)
    member_sample = _sample_indices(
        len(probs_members), MIA_EVAL_PER_HALF, eval_rng
    )
    nonmem_sample = _sample_indices(
        len(probs_nonmem), MIA_EVAL_PER_HALF, eval_rng
    )
    eval_probs = np.concatenate(
        [probs_members[member_sample], probs_nonmem[nonmem_sample]]
    )
    eval_y = np.concatenate(
        [
            y[splits.victim_members][member_sample],
            y[splits.victim_nonmembers][nonmem_sample],
        ]
    )
    eval_member_mask = np.concatenate(
        [
            np.ones(len(member_sample), dtype=bool),
            np.zeros(len(nonmem_sample), dtype=bool),
        ]
    )

    yeom_result = None
    if config.run_yeom:
        yeom_result = run_yeom_attack(eval_y, eval_probs, eval_member_mask)

    shokri_result = None
    if config.run_shokri:
        shokri_result = _run_shokri_attack(
            X=X,
            y=y,
            shadow_pool_indices=splits.shadow_pool,
            image_ppts=image_ppts,
            embedding_ppts=embedding_ppts,
            victim_eval_probs=eval_probs,
            eval_member_mask=eval_member_mask,
            config=config,
        )

    return RunResult(
        config=config,
        utility=utility,
        privacy=PrivacyResult(yeom=yeom_result, shokri=shokri_result),
        efficiency=EfficiencyResult(
            train_latency_seconds=float(train_latency),
            inference_latency_ms_per_query=float(inference_latency_ms),
            memory_peak_mb=_peak_memory_mb(),
            embedding_bytes_per_query=EMBEDDING_BYTES_PER_QUERY,
        ),
        encoder_hash_id=config.victim_encoder_hash,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# --- Result persistence ---


def _run_result_to_jsonable(result: RunResult) -> dict:
    """Convert a RunResult to a JSON-serializable dict matching spec §8."""
    config_dict = {
        "dp_enabled": result.config.dp_enabled,
        "dp_epsilon": result.config.dp_epsilon,
        "bie_enabled": result.config.bie_enabled,
        "bie_key_seed": result.config.bie_key_seed,
        "bie_tile_size": result.config.bie_tile_size,
        "smpc_enabled": result.config.smpc_enabled,
        "smpc_shares": result.config.smpc_shares,
        "epochs": result.config.epochs,
        "seed": result.config.seed,
    }
    privacy_dict = {
        "yeom": (
            None
            if result.privacy.yeom is None
            else {
                "attack_accuracy": result.privacy.yeom.attack_accuracy,
                "attack_auc": result.privacy.yeom.attack_auc,
                "threshold": result.privacy.yeom.threshold,
            }
        ),
        "shokri": (
            None
            if result.privacy.shokri is None
            else {
                "attack_accuracy": result.privacy.shokri.attack_accuracy,
                "attack_auc": result.privacy.shokri.attack_auc,
            }
        ),
    }
    return {
        "tag": result.config.tag,
        "config": config_dict,
        "encoder_hash": result.encoder_hash_id,
        "utility": result.utility._asdict(),
        "privacy": privacy_dict,
        "efficiency": result.efficiency._asdict(),
        "timestamp": result.timestamp,
    }


def append_run_result(result: RunResult, runs_jsonl_path: Path) -> None:
    """Append one JSON line to results/runs.jsonl (creates parent dir)."""
    runs_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    payload = _run_result_to_jsonable(result)
    with runs_jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
