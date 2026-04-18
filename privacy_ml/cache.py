"""Encoder + embedding cache — path management and metadata I/O.

Encoders are identified by a 12-hex-digit SHA-1 of their configuration.
When the configuration changes (BIE on/off, key seed, tile size,
training seed, epoch count, or SCHEMA_VERSION bump), the hash changes
and a fresh encoder is trained. Stale caches become unreachable rather than being
silently reused.

This module deliberately does NOT save or load Keras models — that
requires TensorFlow. Model I/O lives in `runner.py`; `cache.py` only
owns paths, hashing, and metadata so it stays testable without TF.

See docs/superpowers/specs/2026-04-18-mia-design.md §5.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, NamedTuple, Tuple

from .schema import SCHEMA_VERSION

ENCODERS_SUBDIR: str = "encoders"
EMBEDDINGS_SUBDIR: str = "embeddings"
ENCODER_WEIGHTS_SUFFIX: str = ".keras"
ENCODER_META_SUFFIX: str = ".meta.json"
EMBEDDING_FILE_SUFFIX: str = ".npy"
HASH_LENGTH: int = 12


class CachePaths(NamedTuple):
    """Canonical paths for a single encoder's cache entry."""

    encoder_weights: Path
    encoder_meta: Path
    embeddings_dir: Path


def encoder_hash(
    bie_on: bool,
    bie_key_seed: int,
    bie_tile_size: int,
    training_seed: int,
    epochs: int,
) -> str:
    """Compute a stable hash of the encoder's configuration.

    The hash is the leading 12 hex digits of SHA-1 over a pipe-separated
    canonical string. Different arguments produce different hashes;
    identical arguments (in the same SCHEMA_VERSION) produce the same
    hash every time.

    Parameters
    ----------
    bie_tile_size
        Tile edge length for ``BlockWiseImageEncryption`` (must divide 150
        when ``bie_on`` is True). When ``bie_on`` is False, pass ``0``
        (ignored for training, but still part of the hash for a stable
        call signature).
    """
    payload = (
        f"bie_on={bie_on}|"
        f"bie_key_seed={bie_key_seed}|"
        f"bie_tile_size={bie_tile_size}|"
        f"training_seed={training_seed}|"
        f"epochs={epochs}|"
        f"schema_version={SCHEMA_VERSION}"
    )
    digest = hashlib.sha1(
        payload.encode("utf-8"),
        usedforsecurity=False,
    ).hexdigest()
    return digest[:HASH_LENGTH]


def cache_paths(cache_root: Path, hash_id: str) -> CachePaths:
    """Derive the canonical paths for one encoder's cache entry."""
    encoder_dir = cache_root / ENCODERS_SUBDIR
    embeddings_dir = cache_root / EMBEDDINGS_SUBDIR / hash_id
    return CachePaths(
        encoder_weights=encoder_dir / f"{hash_id}{ENCODER_WEIGHTS_SUFFIX}",
        encoder_meta=encoder_dir / f"{hash_id}{ENCODER_META_SUFFIX}",
        embeddings_dir=embeddings_dir,
    )


def write_encoder_meta(meta_path: Path, config: Dict[str, Any]) -> None:
    """Persist the configuration that produced an encoder hash."""
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(config, sort_keys=True, indent=2))


def read_encoder_meta(meta_path: Path) -> Dict[str, Any]:
    """Load a cached encoder's configuration record."""
    return json.loads(meta_path.read_text())


def is_encoder_cached(paths: CachePaths) -> bool:
    """Return True iff both the encoder weights and its meta.json exist."""
    return paths.encoder_weights.exists() and paths.encoder_meta.exists()


def embedding_path(paths: CachePaths, split_name: str) -> Path:
    """Return the expected .npy path for one split's cached embeddings."""
    return paths.embeddings_dir / f"{split_name}{EMBEDDING_FILE_SUFFIX}"


def are_embeddings_cached(
    paths: CachePaths,
    split_names: Tuple[str, ...],
) -> bool:
    """Return True iff every named split has a .npy file in the cache."""
    if not paths.embeddings_dir.is_dir():
        return False
    return all(
        embedding_path(paths, name).exists() for name in split_names
    )
