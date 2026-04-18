"""Tests for the runner + CLI.

TF-free suite: exercises RunConfig construction, encoder-hash derivation,
CLI parsing, PPT fallback semantics, and JSON serialization. The
end-to-end smoke test (that actually trains models) lives at the bottom
under pytest.importorskip and runs in Colab.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import List
from unittest.mock import patch

import numpy as np
import pytest

from privacy_ml.ppt.stubs import IdentityEmbedding, IdentityImage
from privacy_ml.run import (
    _DEFAULT_TAG,
    _construct_ppts,
    _load_ppt_class_or_stub,
    _parse_mia_variants,
    build_run_config,
)
from privacy_ml.runner import (
    EMBEDDING_BYTES_PER_QUERY,
    EfficiencyResult,
    PrivacyResult,
    RunConfig,
    RunResult,
    UtilityResult,
    _fit_embedding_ppts,
    _fit_image_ppts,
    _run_result_to_jsonable,
    _sample_indices,
    append_run_result,
)


def _make_config(**overrides) -> RunConfig:
    """Build a RunConfig with sensible test defaults; override per-test."""
    base = dict(
        dp_enabled=False,
        dp_epsilon=1.0,
        bie_enabled=False,
        bie_key_seed=0,
        bie_tile_size=10,
        smpc_enabled=False,
        smpc_shares=2,
        run_yeom=True,
        run_shokri=False,
        epochs=10,
        seed=42,
        data_dir=Path("/tmp/data"),
        cache_dir=Path("/tmp/cache"),
        output_dir=Path("/tmp/results"),
        tag="test",
    )
    base.update(overrides)
    return RunConfig(**base)


# --- RunConfig ---


def test_victim_encoder_hash_is_stable() -> None:
    config = _make_config()
    assert config.victim_encoder_hash == _make_config().victim_encoder_hash


def test_victim_and_shadow_hashes_differ() -> None:
    config = _make_config()
    assert config.victim_encoder_hash != config.shadow_encoder_hash(0)


def test_shadow_hashes_differ_across_indices() -> None:
    config = _make_config()
    assert config.shadow_encoder_hash(0) != config.shadow_encoder_hash(1)


def test_bie_flag_changes_encoder_hash() -> None:
    no_bie = _make_config(bie_enabled=False)
    with_bie = _make_config(bie_enabled=True)
    assert no_bie.victim_encoder_hash != with_bie.victim_encoder_hash


def test_epochs_changes_encoder_hash() -> None:
    a = _make_config(epochs=10)
    b = _make_config(epochs=11)
    assert a.victim_encoder_hash != b.victim_encoder_hash


def test_bie_tile_size_changes_hash_when_bie_enabled() -> None:
    a = _make_config(bie_enabled=True, bie_tile_size=10)
    b = _make_config(bie_enabled=True, bie_tile_size=15)
    assert a.victim_encoder_hash != b.victim_encoder_hash


def test_bie_tile_size_is_irrelevant_when_bie_disabled() -> None:
    # When BIE is off, tile_size must not affect the cache key; otherwise
    # every baseline would re-train per tile_size CLI default change.
    a = _make_config(bie_enabled=False, bie_tile_size=10)
    b = _make_config(bie_enabled=False, bie_tile_size=15)
    assert a.victim_encoder_hash == b.victim_encoder_hash


# --- CLI parsing ---


def test_parse_mia_variants_both() -> None:
    assert _parse_mia_variants("yeom,shokri") == (True, True)


def test_parse_mia_variants_yeom_only() -> None:
    assert _parse_mia_variants("yeom") == (True, False)


def test_parse_mia_variants_shokri_only() -> None:
    assert _parse_mia_variants("shokri") == (False, True)


def test_parse_mia_variants_rejects_unknown() -> None:
    with pytest.raises(SystemExit):
        _parse_mia_variants("yeom,bogus")


def test_parse_mia_variants_rejects_empty() -> None:
    with pytest.raises(SystemExit):
        _parse_mia_variants("")


def test_build_run_config_maps_all_flags() -> None:
    from argparse import Namespace

    ns = Namespace(
        dp=True,
        dp_epsilon=0.5,
        bie=True,
        bie_key_seed=7,
        bie_tile_size=15,
        smpc=True,
        smpc_shares=3,
        mia="yeom,shokri",
        epochs=5,
        seed=99,
        data_dir=Path("/a"),
        cache_dir=Path("/b"),
        output_dir=Path("/c"),
        tag="weird-tag",
    )
    config = build_run_config(ns)
    assert config.dp_enabled is True
    assert config.dp_epsilon == 0.5
    assert config.bie_enabled is True
    assert config.bie_key_seed == 7
    assert config.bie_tile_size == 15
    assert config.smpc_enabled is True
    assert config.smpc_shares == 3
    assert config.run_yeom is True
    assert config.run_shokri is True
    assert config.epochs == 5
    assert config.seed == 99
    assert config.tag == "weird-tag"


# --- PPT fallback ---


def test_load_ppt_class_or_stub_returns_stub_on_missing_module() -> None:
    # privacy_ml.ppt.dp doesn't exist yet — this is the exact case we guard for
    with pytest.warns(UserWarning):
        cls = _load_ppt_class_or_stub(
            "privacy_ml.ppt.nonexistent_module",
            "WhateverClass",
            IdentityEmbedding,
            flag_name="--test",
        )
    assert cls is IdentityEmbedding


def test_load_ppt_class_or_stub_returns_stub_when_class_missing() -> None:
    with pytest.warns(UserWarning):
        cls = _load_ppt_class_or_stub(
            "privacy_ml.ppt.stubs",
            "NoSuchClassInModule",
            IdentityEmbedding,
            flag_name="--test",
        )
    assert cls is IdentityEmbedding


def test_load_ppt_class_or_stub_returns_real_class_when_present() -> None:
    cls = _load_ppt_class_or_stub(
        "privacy_ml.ppt.stubs",
        "IdentityEmbedding",
        IdentityImage,  # wrong stub on purpose, to verify we pick the real class
        flag_name="--test",
    )
    assert cls is IdentityEmbedding


def test_construct_ppts_assigns_layers_correctly() -> None:
    # All three teammate modules (dp, bie, smpc) now ship real classes;
    # the stub-fallback warning is separately exercised by
    # test_load_ppt_class_or_stub_returns_stub_on_missing_module.
    from argparse import Namespace

    ns = Namespace(
        dp=True,
        dp_epsilon=1.0,
        bie=True,
        bie_key_seed=0,
        bie_tile_size=10,
        smpc=True,
        smpc_shares=2,
        seed=42,
    )
    image_ppts, embedding_ppts = _construct_ppts(ns)
    assert len(image_ppts) == 1
    assert len(embedding_ppts) == 2
    assert all(p.layer == "image" for p in image_ppts)
    assert all(p.layer == "embedding" for p in embedding_ppts)


def test_construct_ppts_empty_when_no_flags() -> None:
    from argparse import Namespace

    ns = Namespace(
        dp=False,
        dp_epsilon=1.0,
        bie=False,
        bie_key_seed=0,
        bie_tile_size=10,
        smpc=False,
        smpc_shares=2,
        seed=42,
    )
    image_ppts, embedding_ppts = _construct_ppts(ns)
    assert image_ppts == []
    assert embedding_ppts == []


# --- Helpers ---


def test_sample_indices_shrinks_when_requesting_fewer() -> None:
    rng = np.random.default_rng(0)
    idx = _sample_indices(total=100, n_samples=10, rng=rng)
    assert len(idx) == 10
    assert len(np.unique(idx)) == 10


def test_sample_indices_returns_all_when_requesting_too_many() -> None:
    rng = np.random.default_rng(0)
    idx = _sample_indices(total=5, n_samples=10, rng=rng)
    assert np.array_equal(idx, np.arange(5))


def test_fit_image_ppts_rejects_embedding_layer_ppt() -> None:
    embed_stub = IdentityEmbedding()
    with pytest.raises(ValueError):
        _fit_image_ppts(np.zeros((2, 10, 10, 1)), [embed_stub])


def test_fit_embedding_ppts_rejects_image_layer_ppt() -> None:
    image_stub = IdentityImage()
    with pytest.raises(ValueError):
        _fit_embedding_ppts(np.zeros((2, 128)), [image_stub])


# --- Result serialization ---


def _make_run_result_fixture() -> RunResult:
    from privacy_ml.attacks.shokri import ShokriAttackResult
    from privacy_ml.attacks.yeom import YeomAttackResult

    return RunResult(
        config=_make_config(tag="fixture"),
        utility=UtilityResult(test_accuracy=0.75, f1=0.80, ece=0.05),
        privacy=PrivacyResult(
            yeom=YeomAttackResult(
                attack_accuracy=0.58,
                attack_auc=0.62,
                threshold=0.42,
            ),
            shokri=ShokriAttackResult(attack_accuracy=0.55, attack_auc=0.59),
        ),
        efficiency=EfficiencyResult(
            train_latency_seconds=12.3,
            inference_latency_ms_per_query=0.9,
            memory_peak_mb=256.0,
            embedding_bytes_per_query=EMBEDDING_BYTES_PER_QUERY,
        ),
        encoder_hash_id="abc123def456",
        timestamp="2026-04-18T17:00:00+00:00",
    )


def test_run_result_to_jsonable_contains_all_top_level_keys() -> None:
    result = _make_run_result_fixture()
    payload = _run_result_to_jsonable(result)
    expected_top_keys = {
        "tag",
        "config",
        "encoder_hash",
        "utility",
        "privacy",
        "efficiency",
        "timestamp",
    }
    assert set(payload.keys()) == expected_top_keys


def test_run_result_to_jsonable_reports_none_for_absent_attacks() -> None:
    result = _make_run_result_fixture()
    result = RunResult(
        **{**result.__dict__, "privacy": PrivacyResult(yeom=None, shokri=None)}
    )
    payload = _run_result_to_jsonable(result)
    assert payload["privacy"]["yeom"] is None
    assert payload["privacy"]["shokri"] is None


def test_run_result_to_jsonable_is_json_encodable() -> None:
    result = _make_run_result_fixture()
    payload = _run_result_to_jsonable(result)
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    assert decoded["utility"]["test_accuracy"] == 0.75


def test_append_run_result_creates_jsonl_file(tmp_path: Path) -> None:
    result = _make_run_result_fixture()
    jsonl_path = tmp_path / "nested" / "runs.jsonl"
    append_run_result(result, jsonl_path)
    assert jsonl_path.exists()
    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 1
    decoded = json.loads(lines[0])
    assert decoded["tag"] == "fixture"


def test_append_run_result_is_append_only(tmp_path: Path) -> None:
    result = _make_run_result_fixture()
    jsonl_path = tmp_path / "runs.jsonl"
    append_run_result(result, jsonl_path)
    append_run_result(result, jsonl_path)
    lines = jsonl_path.read_text().strip().splitlines()
    assert len(lines) == 2
