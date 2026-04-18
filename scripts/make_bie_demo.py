"""Generate a clean side-by-side BIE demonstration figure for the paper.

Usage:
    python scripts/make_bie_demo.py \
        --image path/to/any_xray.jpeg \
        --output results/figures/fig_bie_example.png

Loads a chest X-ray, resizes to 150x150 grayscale, applies our canonical
BIE (tile_size=10, key_seed=7 — the same parameters used in the eval
matrix), and writes a two-panel PNG with titled subplots: (a) original,
(b) BIE-permuted. No external watermarks.

Designed to be run in the Colab environment where the dataset is
available, or locally on any representative chest X-ray image.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from privacy_ml.ppt.bie import BlockWiseImageEncryption

IMG_SIDE: int = 150


def load_grayscale_150(path: Path) -> np.ndarray:
    image = Image.open(path).convert("L").resize((IMG_SIDE, IMG_SIDE), Image.BICUBIC)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return array[np.newaxis, ..., np.newaxis]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True,
                        help="Path to a single chest X-ray image (any format PIL can read)")
    parser.add_argument("--output", type=Path,
                        default=Path("results/figures/fig_bie_example.png"))
    parser.add_argument("--tile-size", type=int, default=10)
    parser.add_argument("--key-seed", type=int, default=7)
    args = parser.parse_args()

    if not args.image.exists():
        raise SystemExit(f"Input image not found: {args.image}")

    X = load_grayscale_150(args.image)
    bie = BlockWiseImageEncryption(tile_size=args.tile_size, key_seed=args.key_seed)
    X_bie = bie.apply(X)

    original = X[0, :, :, 0]
    permuted = X_bie[0, :, :, 0]

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.8))
    axes[0].imshow(original, cmap="gray", vmin=0.0, vmax=1.0)
    axes[0].set_title("(a) Original chest X-ray", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(permuted, cmap="gray", vmin=0.0, vmax=1.0)
    axes[1].set_title(
        f"(b) BIE output "
        f"(tile={args.tile_size}, key={args.key_seed})",
        fontsize=11,
    )
    axes[1].axis("off")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
