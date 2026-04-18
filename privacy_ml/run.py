"""CLI entry point: python -m privacy_ml.run --dp --bie --tag my-run

Composes PrivacyMechanism instances from CLI flags and hands them to
the runner. If a teammate hasn't shipped their PPT module yet, the
CLI falls back to an identity stub and prints a warning so work isn't
blocked on missing modules.

See docs/superpowers/specs/2026-04-18-mia-design.md §7.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import List, Tuple

from .ppt.base import PrivacyMechanism
from .ppt.stubs import IdentityEmbedding, IdentityImage
from .runner import RunConfig, append_run_result, run_single_config

DEFAULT_DATA_DIR: Path = Path("./data/chest_xray")
DEFAULT_CACHE_DIR: Path = Path("./results/cache")
DEFAULT_OUTPUT_DIR: Path = Path("./results")
DEFAULT_RUNS_JSONL: str = "runs.jsonl"

_DEFAULT_DP_EPSILON: float = 1.0
_DEFAULT_BIE_KEY_SEED: int = 0
_DEFAULT_BIE_TILE_SIZE: int = 10
_DEFAULT_SMPC_SHARES: int = 2
_DEFAULT_EPOCHS: int = 10
_DEFAULT_SEED: int = 42
_DEFAULT_MIA: str = "yeom,shokri"
_DEFAULT_TAG: str = "run"


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m privacy_ml.run",
        description=(
            "Run the split-model pneumonia-classification pipeline "
            "under a chosen PPT configuration and measure MIA risk."
        ),
    )
    parser.add_argument("--dp", action="store_true", help="Enable Differential Privacy on embeddings.")
    parser.add_argument("--dp-epsilon", type=float, default=_DEFAULT_DP_EPSILON)
    parser.add_argument("--bie", action="store_true", help="Enable Block-wise Image Encryption on images.")
    parser.add_argument("--bie-key-seed", type=int, default=_DEFAULT_BIE_KEY_SEED)
    parser.add_argument("--bie-tile-size", type=int, default=_DEFAULT_BIE_TILE_SIZE)
    parser.add_argument("--smpc", action="store_true", help="Enable SMPC additive secret sharing on embeddings.")
    parser.add_argument("--smpc-shares", type=int, default=_DEFAULT_SMPC_SHARES)
    parser.add_argument("--mia", type=str, default=_DEFAULT_MIA, help="Comma-separated MIA variants (yeom,shokri).")
    parser.add_argument("--epochs", type=int, default=_DEFAULT_EPOCHS)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", type=str, default=_DEFAULT_TAG)
    return parser


def _parse_mia_variants(mia_flag: str) -> Tuple[bool, bool]:
    """Turn --mia 'yeom,shokri' into (run_yeom, run_shokri)."""
    variants = {v.strip().lower() for v in mia_flag.split(",") if v.strip()}
    unknown = variants - {"yeom", "shokri"}
    if unknown:
        raise SystemExit(f"Unknown MIA variants: {unknown}. Expected: yeom,shokri.")
    if not variants:
        raise SystemExit("--mia requires at least one of: yeom, shokri.")
    return ("yeom" in variants, "shokri" in variants)


def _load_ppt_class_or_stub(
    module_path: str,
    class_name: str,
    stub: PrivacyMechanism,
    flag_name: str,
):
    """Try to import a teammate's PPT class; fall back to the identity stub with a warning.

    Lets Egehan run the pipeline end-to-end before Onur/İrem Damla/Eray
    have shipped their real modules. When real modules land, no CLI or
    runner changes are needed — the import just starts succeeding.
    """
    try:
        module = __import__(module_path, fromlist=[class_name])
    except ImportError as e:
        warnings.warn(
            f"[{flag_name}] module '{module_path}' not importable ({e}); "
            f"falling back to IdentityStub. Results will NOT reflect real {flag_name}."
        )
        return stub
    try:
        return getattr(module, class_name)
    except AttributeError:
        warnings.warn(
            f"[{flag_name}] class '{class_name}' missing from '{module_path}'; "
            f"falling back to IdentityStub."
        )
        return stub


def _construct_ppts(args: argparse.Namespace) -> Tuple[List[PrivacyMechanism], List[PrivacyMechanism]]:
    """Instantiate image-layer and embedding-layer PPT lists from CLI flags."""
    image_ppts: List[PrivacyMechanism] = []
    embedding_ppts: List[PrivacyMechanism] = []

    if args.bie:
        BieClass = _load_ppt_class_or_stub(
            "privacy_ml.ppt.bie",
            "BlockWiseImageEncryption",
            IdentityImage,
            flag_name="--bie",
        )
        if BieClass is IdentityImage:
            image_ppts.append(IdentityImage())
        else:
            image_ppts.append(
                BieClass(tile_size=args.bie_tile_size, key_seed=args.bie_key_seed)
            )

    if args.dp:
        DpClass = _load_ppt_class_or_stub(
            "privacy_ml.ppt.dp",
            "DifferentialPrivacy",
            IdentityEmbedding,
            flag_name="--dp",
        )
        if DpClass is IdentityEmbedding:
            embedding_ppts.append(IdentityEmbedding())
        else:
            embedding_ppts.append(
                DpClass(epsilon=args.dp_epsilon, sensitivity=1.0, seed=args.seed)
            )

    if args.smpc:
        SmpcClass = _load_ppt_class_or_stub(
            "privacy_ml.ppt.smpc",
            "SecretShareSMPC",
            IdentityEmbedding,
            flag_name="--smpc",
        )
        if SmpcClass is IdentityEmbedding:
            embedding_ppts.append(IdentityEmbedding())
        else:
            embedding_ppts.append(
                SmpcClass(n_shares=args.smpc_shares, seed=args.seed)
            )

    return image_ppts, embedding_ppts


def build_run_config(args: argparse.Namespace) -> RunConfig:
    """Turn an argparse Namespace into a RunConfig dataclass."""
    run_yeom, run_shokri = _parse_mia_variants(args.mia)
    return RunConfig(
        dp_enabled=args.dp,
        dp_epsilon=float(args.dp_epsilon),
        bie_enabled=args.bie,
        bie_key_seed=int(args.bie_key_seed),
        bie_tile_size=int(args.bie_tile_size),
        smpc_enabled=args.smpc,
        smpc_shares=int(args.smpc_shares),
        run_yeom=run_yeom,
        run_shokri=run_shokri,
        epochs=int(args.epochs),
        seed=int(args.seed),
        data_dir=Path(args.data_dir),
        cache_dir=Path(args.cache_dir),
        output_dir=Path(args.output_dir),
        tag=str(args.tag),
    )


def main(argv: List[str]) -> int:
    args = _build_argparser().parse_args(argv)
    config = build_run_config(args)
    image_ppts, embedding_ppts = _construct_ppts(args)
    result = run_single_config(config, image_ppts, embedding_ppts)
    runs_jsonl = config.output_dir / DEFAULT_RUNS_JSONL
    append_run_result(result, runs_jsonl)
    print(f"[{config.tag}] done. encoder_hash={result.encoder_hash_id}; "
          f"test_acc={result.utility.test_accuracy:.3f}; "
          f"yeom_acc={result.privacy.yeom.attack_accuracy if result.privacy.yeom else 'n/a'}; "
          f"shokri_acc={result.privacy.shokri.attack_accuracy if result.privacy.shokri else 'n/a'}; "
          f"appended to {runs_jsonl}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
