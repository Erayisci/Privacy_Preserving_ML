"""Generate publication-ready figures from results/runs.jsonl.

Run: python3 scripts/plot_results.py [--input PATH] [--output-dir PATH]

Produces into results/figures/:
  - fig_utility_privacy.png   grouped bar: test_acc / yeom / shokri per config
  - fig_reconstruction.png    bar: recon_psnr per config (lower = better defense)
  - fig_pareto.png            scatter: utility vs MIA attack accuracy
  - fig_summary_table.png     rendered summary table
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


CONFIG_ORDER: List[str] = [
    "baseline",
    "dp-only",
    "bie-only",
    "smpc-only",
    "dp+bie",
    "dp+smpc",
    "bie+smpc",
    "all-three",
]

DP_COLOR = "#d62728"
NO_DP_COLOR = "#1f77b4"


def load_runs(path: Path) -> Dict[str, dict]:
    records: Dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            records[rec["tag"]] = rec
    return records


def ordered_tags(records: Dict[str, dict]) -> List[str]:
    return [t for t in CONFIG_ORDER if t in records]


def _bar_colors(tags: List[str]) -> List[str]:
    return [DP_COLOR if "dp" in t or t == "all-three" else NO_DP_COLOR for t in tags]


def plot_utility_privacy(records: Dict[str, dict], out_path: Path) -> None:
    tags = ordered_tags(records)
    test_acc = [records[t]["utility"]["test_accuracy"] for t in tags]
    yeom = [records[t]["privacy"]["yeom"]["attack_accuracy"] for t in tags]
    shokri = [records[t]["privacy"]["shokri"]["attack_accuracy"] for t in tags]

    x = np.arange(len(tags))
    width = 0.27

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width, test_acc, width, label="Test accuracy (utility)", color="#2ca02c")
    ax.bar(x, yeom, width, label="Yeom MIA accuracy", color="#ff7f0e")
    ax.bar(x + width, shokri, width, label="Shokri MIA accuracy", color="#9467bd")

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Random MIA baseline")
    ax.set_xticks(x)
    ax.set_xticklabels(tags, rotation=20, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Utility vs MIA Attack Accuracy Across PPT Configurations")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_reconstruction(records: Dict[str, dict], out_path: Path) -> None:
    tags = ordered_tags(records)
    psnr = [records[t]["reconstruction"]["psnr"] for t in tags]
    colors = _bar_colors(tags)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(tags, psnr, color=colors, edgecolor="black", linewidth=0.5)

    baseline_psnr = records["baseline"]["reconstruction"]["psnr"]
    ax.axhline(baseline_psnr, color="gray", linestyle="--", linewidth=0.8,
               label=f"Baseline PSNR ({baseline_psnr:.1f} dB)")

    for bar, value in zip(bars, psnr):
        ax.annotate(f"{value:.1f}",
                    xy=(bar.get_x() + bar.get_width() / 2, value),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=9)

    ax.set_ylabel("Reconstruction PSNR (dB)  —  lower = stronger defense")
    ax.set_title("Reconstruction Attack: PSNR Per Configuration")
    ax.set_xticklabels(tags, rotation=20, ha="right")
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=DP_COLOR, label="Config includes DP"),
        plt.Rectangle((0, 0), 1, 1, color=NO_DP_COLOR, label="No DP"),
    ]
    ax.legend(handles=legend_handles + ax.get_legend_handles_labels()[0],
              loc="lower right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_pareto(records: Dict[str, dict], out_path: Path) -> None:
    tags = ordered_tags(records)
    test_acc = [records[t]["utility"]["test_accuracy"] for t in tags]
    yeom = [records[t]["privacy"]["yeom"]["attack_accuracy"] for t in tags]
    colors = _bar_colors(tags)

    fig, ax = plt.subplots(figsize=(8, 6))
    for tag, acc, atk, color in zip(tags, test_acc, yeom, colors):
        ax.scatter(acc, atk, s=160, color=color, edgecolor="black", linewidth=0.8, zorder=3)
        ax.annotate(tag, (acc, atk), xytext=(7, 4), textcoords="offset points", fontsize=10)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8,
               label="Random guess (perfect defense)")

    ax.set_xlabel("Test accuracy (utility)  →  higher is better")
    ax.set_ylabel("Yeom MIA accuracy  →  lower is better")
    ax.set_title("Utility vs Privacy Trade-off (Yeom MIA)")
    ax.grid(alpha=0.3)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=DP_COLOR,
                   markeredgecolor="black", markersize=11, label="DP on"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=NO_DP_COLOR,
                   markeredgecolor="black", markersize=11, label="DP off"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.95)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_summary_table(records: Dict[str, dict], out_path: Path) -> None:
    tags = ordered_tags(records)
    columns = ["test_acc", "f1", "yeom", "shokri", "recon_psnr", "recon_ssim"]
    rows: List[List[str]] = []
    for t in tags:
        r = records[t]
        rows.append([
            f"{r['utility']['test_accuracy']:.3f}",
            f"{r['utility']['f1']:.3f}",
            f"{r['privacy']['yeom']['attack_accuracy']:.3f}",
            f"{r['privacy']['shokri']['attack_accuracy']:.3f}",
            f"{r['reconstruction']['psnr']:.2f}",
            f"{r['reconstruction']['ssim']:.3f}",
        ])

    fig, ax = plt.subplots(figsize=(10, 0.55 * (len(tags) + 2)))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        rowLabels=tags,
        colLabels=columns,
        cellLoc="center",
        rowLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.5)

    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0 or col_idx == -1:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#e0e0e0")

    ax.set_title("Summary: Utility + Privacy + Reconstruction Per Configuration",
                 pad=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path,
                        default=Path("runs.jsonl"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("results/figures"))
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Input not found: {args.input}")

    records = load_runs(args.input)
    missing = [t for t in CONFIG_ORDER if t not in records]
    if missing:
        print(f"WARN: missing configs (will be skipped): {missing}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    plot_utility_privacy(records, args.output_dir / "fig_utility_privacy.png")
    plot_reconstruction(records, args.output_dir / "fig_reconstruction.png")
    plot_pareto(records, args.output_dir / "fig_pareto.png")
    plot_summary_table(records, args.output_dir / "fig_summary_table.png")

    print(f"Wrote 4 figures to {args.output_dir.resolve()}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
