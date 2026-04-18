"""Tests for the encoder+embedding cache's path management and hashing.

Model I/O is not exercised here (that's TF-dependent and lives in
runner.py); this suite verifies hash stability, path derivation,
metadata round-trips, and existence checks.
"""
from __future__ import annotations

import pytest

from privacy_ml.cache import (
    EMBEDDING_FILE_SUFFIX,
    ENCODER_META_SUFFIX,
    ENCODER_WEIGHTS_SUFFIX,
    ENCODERS_SUBDIR,
    EMBEDDINGS_SUBDIR,
    HASH_LENGTH,
    are_embeddings_cached,
    cache_paths,
    embedding_path,
    encoder_hash,
    is_encoder_cached,
    read_encoder_meta,
    write_encoder_meta,
)

_REF_CONFIG = dict(
    bie_on=False,
    bie_key_seed=0,
    bie_tile_size=0,
    training_seed=42,
    epochs=10,
)


def test_encoder_hash_is_deterministic() -> None:
    first = encoder_hash(**_REF_CONFIG)
    second = encoder_hash(**_REF_CONFIG)
    assert first == second


def test_encoder_hash_length_and_alphabet() -> None:
    h = encoder_hash(**_REF_CONFIG)
    assert len(h) == HASH_LENGTH
    assert all(c in "0123456789abcdef" for c in h)


@pytest.mark.parametrize(
    "field,new_value",
    [
        ("bie_on", True),
        ("bie_key_seed", 1),
        ("bie_tile_size", 10),
        ("training_seed", 43),
        ("epochs", 11),
    ],
)
def test_encoder_hash_changes_with_any_field(field: str, new_value) -> None:
    baseline = encoder_hash(**_REF_CONFIG)
    perturbed = {**_REF_CONFIG, field: new_value}
    assert encoder_hash(**perturbed) != baseline


def test_cache_paths_points_at_canonical_layout(tmp_path) -> None:
    paths = cache_paths(tmp_path, hash_id="abc123def456")
    assert paths.encoder_weights == (
        tmp_path / ENCODERS_SUBDIR / f"abc123def456{ENCODER_WEIGHTS_SUFFIX}"
    )
    assert paths.encoder_meta == (
        tmp_path / ENCODERS_SUBDIR / f"abc123def456{ENCODER_META_SUFFIX}"
    )
    assert paths.embeddings_dir == (
        tmp_path / EMBEDDINGS_SUBDIR / "abc123def456"
    )


def test_embedding_path_uses_npy_suffix(tmp_path) -> None:
    paths = cache_paths(tmp_path, hash_id="abc")
    path = embedding_path(paths, "victim_members")
    assert path == paths.embeddings_dir / f"victim_members{EMBEDDING_FILE_SUFFIX}"


def test_write_and_read_encoder_meta_round_trip(tmp_path) -> None:
    meta_path = tmp_path / "meta.json"
    config = {"bie_on": False, "epochs": 10, "training_seed": 42}
    write_encoder_meta(meta_path, config)
    assert meta_path.exists()
    assert read_encoder_meta(meta_path) == config


def test_write_encoder_meta_creates_parent_dirs(tmp_path) -> None:
    meta_path = tmp_path / "deep" / "nested" / "meta.json"
    write_encoder_meta(meta_path, {"k": "v"})
    assert meta_path.exists()


def test_is_encoder_cached_false_when_nothing_exists(tmp_path) -> None:
    paths = cache_paths(tmp_path, hash_id="none")
    assert not is_encoder_cached(paths)


def test_is_encoder_cached_false_when_only_weights_exist(tmp_path) -> None:
    paths = cache_paths(tmp_path, hash_id="partial")
    paths.encoder_weights.parent.mkdir(parents=True, exist_ok=True)
    paths.encoder_weights.touch()
    assert not is_encoder_cached(paths)


def test_is_encoder_cached_false_when_only_meta_exists(tmp_path) -> None:
    paths = cache_paths(tmp_path, hash_id="partial")
    paths.encoder_meta.parent.mkdir(parents=True, exist_ok=True)
    paths.encoder_meta.touch()
    assert not is_encoder_cached(paths)


def test_is_encoder_cached_true_when_both_exist(tmp_path) -> None:
    paths = cache_paths(tmp_path, hash_id="complete")
    paths.encoder_weights.parent.mkdir(parents=True, exist_ok=True)
    paths.encoder_weights.touch()
    paths.encoder_meta.touch()
    assert is_encoder_cached(paths)


def test_are_embeddings_cached_false_when_dir_missing(tmp_path) -> None:
    paths = cache_paths(tmp_path, hash_id="x")
    assert not are_embeddings_cached(
        paths, ("victim_members", "victim_nonmembers")
    )


def test_are_embeddings_cached_false_when_one_file_missing(tmp_path) -> None:
    paths = cache_paths(tmp_path, hash_id="x")
    paths.embeddings_dir.mkdir(parents=True, exist_ok=True)
    embedding_path(paths, "victim_members").touch()
    assert not are_embeddings_cached(
        paths, ("victim_members", "victim_nonmembers")
    )


def test_are_embeddings_cached_true_when_all_files_present(tmp_path) -> None:
    paths = cache_paths(tmp_path, hash_id="x")
    paths.embeddings_dir.mkdir(parents=True, exist_ok=True)
    embedding_path(paths, "victim_members").touch()
    embedding_path(paths, "victim_nonmembers").touch()
    assert are_embeddings_cached(
        paths, ("victim_members", "victim_nonmembers")
    )


def test_are_embeddings_cached_with_shadow_splits(tmp_path) -> None:
    paths = cache_paths(tmp_path, hash_id="x")
    paths.embeddings_dir.mkdir(parents=True, exist_ok=True)
    names = (
        "victim_members",
        "victim_nonmembers",
        "shadow_0_train",
        "shadow_0_holdout",
    )
    for n in names:
        embedding_path(paths, n).touch()
    assert are_embeddings_cached(paths, names)
