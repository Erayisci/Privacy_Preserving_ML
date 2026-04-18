"""Generate run_in_colab.ipynb from inline cell sources.

Keeping the generator in-repo lets us regenerate the notebook cleanly
whenever a cell changes — avoids hand-editing JSON. Idempotent: runs
produce the same notebook byte-for-byte (no timestamps / execution
counts embedded).

Usage:
    python scripts/generate_colab_notebook.py

Writes to repo root: run_in_colab.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "run_in_colab.ipynb"


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.rstrip("\n").split("\n")],
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.rstrip("\n").split("\n")],
    }


CELL_1_MARKDOWN = """\
# Privacy-Preserving ML — Colab Experiment Driver

CS475 privacy-preserving chest X-ray project — runs the 8-configuration
MIA evaluation matrix from `privacy_ml` package on Colab T4 GPU.

## Before you start

1. **Runtime** → **Change runtime type** → **Hardware accelerator: T4 GPU**
   (free tier is enough — the whole pipeline finishes in 30–45 minutes).
2. Have your **`kaggle.json`** file ready
   (Kaggle → Settings → API → Create New API Token → downloads `kaggle.json`).

## What this notebook does

1. Verify a GPU is attached
2. Clone / pull the latest `main` from the shared repo
3. Install the `privacy_ml` package + test deps
4. Run the test suite (expect all green)
5. Download the `chest-xray-pneumonia` Kaggle dataset
6. Verify dataset structure
7. Run **8 canonical PPT configurations** (baseline, 3 singles, 3 pairs, all-three)
8. Display results as a pandas DataFrame
9. Download `runs.jsonl` to commit to the repo

## Authoritative references

- Design: `docs/superpowers/specs/2026-04-18-mia-design.md`
- Roadmap: `remaining_roadmap.md`
"""


CELL_2_GPU_CHECK = """\
# Cell 2 — GPU sanity check
import subprocess

result = subprocess.run(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    capture_output=True, text=True,
)

if result.returncode != 0 or not result.stdout.strip():
    raise RuntimeError(
        "No GPU detected.\\n"
        "Go to Runtime -> Change runtime type -> Hardware accelerator: T4 GPU, "
        "then 'Save' and re-run this cell."
    )

print(f"GPU available: {result.stdout.strip()}")

import tensorflow as tf
tf_gpus = tf.config.list_physical_devices('GPU')
print(f"TensorFlow sees {len(tf_gpus)} GPU(s): {[g.name for g in tf_gpus]}")
"""


CELL_3_CLONE = """\
# Cell 3 — Clone or sync repo (safe for team workflow)
import os
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/Erayisci/Privacy_Preserving_ML.git"
REPO_PATH = Path("/content/Privacy_Preserving_ML")
BRANCH = "main"


def stream(cmd, cwd=None):
    \"\"\"Run and stream output to the cell; raises on non-zero exit.\"\"\"
    print(f"$ {' '.join(str(c) for c in cmd)}")
    subprocess.check_call(cmd, cwd=cwd)


def capture(cmd, cwd=None):
    \"\"\"Run and return stdout; raises on non-zero exit.\"\"\"
    return subprocess.check_output(cmd, cwd=cwd, text=True)


if REPO_PATH.exists() and (REPO_PATH / ".git").is_dir():
    print(f"Existing clone found; syncing to origin/{BRANCH}...\\n")

    # Teammates push constantly. Local edits in Colab are almost always
    # unintentional (editor autosave, partial notebook runs, etc.) and
    # would block a rebase. Auto-stash them so the pull proceeds; if you
    # actually wanted them, recover via:
    #   !cd /content/Privacy_Preserving_ML && git stash list
    #   !cd /content/Privacy_Preserving_ML && git stash pop
    dirty = capture(["git", "status", "--porcelain"], cwd=REPO_PATH).strip()
    if dirty:
        print("!  Uncommitted changes in Colab - auto-stashing before rebase:")
        print(dirty)
        stream(
            ["git", "stash", "push", "-u", "-m", "colab-autostash"],
            cwd=REPO_PATH,
        )

    stream(["git", "fetch", "origin", BRANCH], cwd=REPO_PATH)
    stream(["git", "pull", "--rebase", "origin", BRANCH], cwd=REPO_PATH)
else:
    print(f"Cloning {REPO_URL} -> {REPO_PATH}...\\n")
    stream(["git", "clone", "--branch", BRANCH, REPO_URL, str(REPO_PATH)])

os.chdir(REPO_PATH)

print("\\n--- Recent commits on main ---")
stream(["git", "log", "--oneline", "-5"])
print("\\n--- Branch state ---")
stream(["git", "status", "--short", "--branch"])
"""


CELL_4_INSTALL = """\
# Cell 4 — Install runtime + test deps and make privacy_ml importable
#
# We intentionally avoid `pip install -e .` here — on Colab (Python 3.12)
# it frequently fails at "Getting requirements to build editable did not
# run successfully" due to a setuptools/PEP 660 build-isolation quirk.
# Direct sys.path injection is simpler, reproducible, and skips the dance.
import subprocess
import sys

print("Installing runtime + test deps...")
print("(TF, numpy, matplotlib, pandas are already in Colab.)\\n")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "scikit-learn>=1.3",
    "pytest>=7.0",
])

# Put the repo on the import path so `import privacy_ml` works without
# an editable install.
REPO_PATH = "/content/Privacy_Preserving_ML"
if REPO_PATH not in sys.path:
    sys.path.insert(0, REPO_PATH)

print("\\nInstall complete. Verifying imports...\\n")

import importlib

# Core modules - Egehan ships these; fail loudly if one is broken.
CORE_MODULES = [
    "privacy_ml",
    "privacy_ml.models",
    "privacy_ml.data",
    "privacy_ml.cache",
    "privacy_ml.metrics",
    "privacy_ml.attacks.yeom",
    "privacy_ml.attacks.shokri",
    "privacy_ml.ppt.base",
    "privacy_ml.ppt.stubs",
    "privacy_ml.runner",
    "privacy_ml.run",
]

# Teammate PPT modules - optional. If missing, run.py falls back to
# IdentityStub with a warning at CLI time; the pipeline still runs.
TEAMMATE_MODULES = [
    ("privacy_ml.ppt.dp",   "Onur (DP)"),
    ("privacy_ml.ppt.bie",  "Irem Damla (BIE)"),
    ("privacy_ml.ppt.smpc", "Eray (SMPC)"),
]

print("Core modules:")
for module_name in CORE_MODULES:
    importlib.import_module(module_name)
    print(f"  OK {module_name}")

print("\\nTeammate PPT modules:")
missing_teammates = []
for module_name, owner in TEAMMATE_MODULES:
    try:
        importlib.import_module(module_name)
        print(f"  OK {module_name}  <- {owner}")
    except ImportError:
        print(f"  !  {module_name} not found  <- {owner} hasn't pushed yet "
              f"(runner will use IdentityStub)")
        missing_teammates.append(owner)

import privacy_ml
print(f"\\nprivacy_ml version: {privacy_ml.__version__}")

if missing_teammates:
    print(f"\\n!  Missing PPT modules from: {', '.join(missing_teammates)}")
    print("   Running now would produce baseline-equivalent results for those flags.")
    print("   That's fine for a smoke test; re-run this notebook once they push.")
"""


CELL_5_PYTEST = """\
# Cell 5 — Run the test suite to verify a clean baseline before experiments
import subprocess
import sys

print("Running pytest on the full test suite...")
print("(Local: non-TF tests + TF-dependent tests in test_models.py")
print(" that only run here on Colab. Plus any teammate tests they've pushed.)\\n")

try:
    subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
        check=True,
    )
    print("\\nAll tests green. Safe to run experiments.")
except subprocess.CalledProcessError:
    print("\\nSome tests failed. Stop and investigate before running experiments.")
    print("   Common causes:")
    print("   - A teammate's PPT module has a bug (check `git log --oneline -10`)")
    print("   - Missing dependency (re-run Cell 4)")
    print("   - GPU / TF version mismatch (re-run Cell 2)")
    print("   - Local edits auto-stashed in Cell 3 - re-run it to sync cleanly")
    raise
"""


CELL_6_DOWNLOAD = """\
# Cell 6 — Download & extract chest-xray-pneumonia dataset
import subprocess
import sys
from pathlib import Path

KAGGLE_JSON_PATH = Path("/root/.kaggle/kaggle.json")
ZIP_PATH = Path("/content/chest-xray-pneumonia.zip")
EXTRACT_DIR = Path("/content/chest_xray")
SENTINEL_DIR = EXTRACT_DIR / "chest_xray" / "train" / "NORMAL"

if SENTINEL_DIR.is_dir():
    print(f"OK Dataset already extracted at {EXTRACT_DIR}")
    print("   (Skipping download/unzip. `rm -rf` the folder to force re-download.)")
else:
    # 1) Get kaggle.json credentials (upload prompt if not already present)
    if not KAGGLE_JSON_PATH.exists():
        print("Upload your kaggle.json")
        print("(Kaggle -> Settings -> Account -> API -> Create New API Token)\\n")
        from google.colab import files
        uploaded = files.upload()
        if "kaggle.json" not in uploaded:
            raise RuntimeError(
                "No kaggle.json uploaded. Get one from "
                "https://www.kaggle.com/settings/account -> API section."
            )
        KAGGLE_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        KAGGLE_JSON_PATH.write_bytes(uploaded["kaggle.json"])
        KAGGLE_JSON_PATH.chmod(0o600)
        print(f"OK Saved credentials to {KAGGLE_JSON_PATH}")
    else:
        print(f"OK Using existing credentials at {KAGGLE_JSON_PATH}")

    # 2) Ensure kaggle CLI is installed (Colab usually has it; cheap check)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "kaggle"]
    )

    # 3) Download the zip (~2.3 GB; typically 20-40s on Colab GPU runtime)
    print("\\nDownloading chest-xray-pneumonia (~2.3 GB)...")
    subprocess.check_call([
        "kaggle", "datasets", "download",
        "-d", "paultimothymooney/chest-xray-pneumonia",
        "-p", "/content",
        "--force",
    ])

    # 4) Unzip
    print("\\nExtracting...")
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.check_call([
        "unzip", "-q", "-o", str(ZIP_PATH),
        "-d", str(EXTRACT_DIR),
    ])
    print(f"OK Extracted to {EXTRACT_DIR}")

# Reminder: the zip extracts to a nested chest_xray/chest_xray/ layout.
# privacy_ml.data.resolve_kaggle_base() handles both flat and nested paths.
"""


CELL_7_VERIFY = """\
# Cell 7 — Verify dataset structure + image counts
from pathlib import Path

from privacy_ml.data import (
    KAGGLE_CLASS_DIRS,
    KAGGLE_SUBDIRS,
    SUPPORTED_IMAGE_SUFFIXES,
    resolve_kaggle_base,
)

RAW_DOWNLOAD_DIR = Path("/content/chest_xray")

# resolve_kaggle_base handles both flat and nested chest_xray/chest_xray/ layouts.
base = resolve_kaggle_base(RAW_DOWNLOAD_DIR)
print(f"Dataset base: {base}\\n")

print(f"{'Split':<8} {'NORMAL':>8} {'PNEUMONIA':>10} {'Total':>8}")
print("-" * 40)

totals = {"NORMAL": 0, "PNEUMONIA": 0}
for subdir in KAGGLE_SUBDIRS:
    row = {}
    for class_name, _ in KAGGLE_CLASS_DIRS:
        class_dir = base / subdir / class_name
        if class_dir.is_dir():
            n = sum(
                1 for p in class_dir.iterdir()
                if p.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            )
            row[class_name] = n
            totals[class_name] += n
        else:
            row[class_name] = None

    normal = row.get("NORMAL")
    pneu = row.get("PNEUMONIA")
    split_total = (normal or 0) + (pneu or 0)
    print(f"{subdir:<8} {str(normal):>8} {str(pneu):>10} {split_total:>8}")

grand_total = totals["NORMAL"] + totals["PNEUMONIA"]
ratio = (
    totals["PNEUMONIA"] / totals["NORMAL"]
    if totals["NORMAL"] > 0
    else float("inf")
)
print("-" * 40)
print(f"{'Total':<8} {totals['NORMAL']:>8} {totals['PNEUMONIA']:>10} {grand_total:>8}")
print(f"\\nClass imbalance pneumonia : normal = {ratio:.2f} : 1  (spec §4 expects ~3:1)")

EXPECTED_MIN = 5000
EXPECTED_MAX = 6500
if grand_total < EXPECTED_MIN or grand_total > EXPECTED_MAX:
    raise RuntimeError(
        f"Total image count {grand_total} is outside expected range "
        f"[{EXPECTED_MIN}, {EXPECTED_MAX}]. Re-run Cell 6 to re-download."
    )

print(f"\\nDataset OK ({grand_total} images; spec expects ~5856)")
"""


CELL_8_SEPARATOR = """\
---

## Experiments

Setup done. The next cell runs the 8-configuration MIA evaluation matrix
(baseline + 3 single PPTs + 3 pairs + all-three). Expected runtime on
a T4 GPU: **30–45 minutes** on the first run, seconds per config on
cached reruns.

Each configuration produces:
- **Utility**: test accuracy, F1, ECE
- **Privacy**: Yeom + Shokri attack accuracy + AUC
- **Efficiency**: training / inference latency, memory, bandwidth

All records are appended to `results/runs.jsonl` — one JSON line per run.
"""


CELL_9_MATRIX = """\
# Cell 9 — Run the 8-configuration MIA evaluation matrix
import time
from pathlib import Path

from privacy_ml.ppt.bie import BlockWiseImageEncryption
from privacy_ml.ppt.dp import DifferentialPrivacy
from privacy_ml.ppt.smpc import SecretShareSMPC
from privacy_ml.runner import RunConfig, append_run_result, run_single_config

# Results live in the repo's results/ dir (gitignored). If you want to
# commit the runs.jsonl afterwards, add it with `git add -f`.
OUTPUT_DIR = Path("/content/Privacy_Preserving_ML/results")
CACHE_DIR = OUTPUT_DIR / "cache"
RUNS_JSONL = OUTPUT_DIR / "runs.jsonl"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Experiment hyperparameters ---
EPOCHS = 10
SEED = 42
DP_EPSILON = 1.0
DP_SENSITIVITY = 1.0
BIE_TILE_SIZE = 10
BIE_KEY_SEED = 7
SMPC_SHARES = 2

# Set FIRST_N to a small number (e.g., 1) for a smoke test.
# None = run all 8 configs.
FIRST_N = None

# 8-config matrix per spec §7.
CONFIGS = [
    ("baseline",  dict(dp=False, bie=False, smpc=False)),
    ("dp-only",   dict(dp=True,  bie=False, smpc=False)),
    ("bie-only",  dict(dp=False, bie=True,  smpc=False)),
    ("smpc-only", dict(dp=False, bie=False, smpc=True)),
    ("dp+bie",    dict(dp=True,  bie=True,  smpc=False)),
    ("dp+smpc",   dict(dp=True,  bie=False, smpc=True)),
    ("bie+smpc",  dict(dp=False, bie=True,  smpc=True)),
    ("all-three", dict(dp=True,  bie=True,  smpc=True)),
]


def _fmt_duration(seconds: float) -> str:
    if seconds >= 60:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    return f"{seconds:.1f}s"


def build_config(tag: str, flags: dict) -> RunConfig:
    return RunConfig(
        dp_enabled=flags["dp"],
        dp_epsilon=DP_EPSILON,
        bie_enabled=flags["bie"],
        bie_key_seed=BIE_KEY_SEED,
        bie_tile_size=BIE_TILE_SIZE,
        smpc_enabled=flags["smpc"],
        smpc_shares=SMPC_SHARES,
        run_yeom=True,
        run_shokri=True,
        epochs=EPOCHS,
        seed=SEED,
        data_dir=base,
        cache_dir=CACHE_DIR,
        output_dir=OUTPUT_DIR,
        tag=tag,
    )


def build_ppts(flags: dict):
    image_ppts, embedding_ppts = [], []
    if flags["bie"]:
        image_ppts.append(
            BlockWiseImageEncryption(
                tile_size=BIE_TILE_SIZE, key_seed=BIE_KEY_SEED
            )
        )
    if flags["dp"]:
        embedding_ppts.append(
            DifferentialPrivacy(
                epsilon=DP_EPSILON,
                sensitivity=DP_SENSITIVITY,
                seed=SEED,
            )
        )
    if flags["smpc"]:
        embedding_ppts.append(
            SecretShareSMPC(n_shares=SMPC_SHARES, seed=SEED)
        )
    return image_ppts, embedding_ppts


configs_to_run = CONFIGS if FIRST_N is None else CONFIGS[:FIRST_N]
print(f"Running {len(configs_to_run)} configuration(s). Output -> {RUNS_JSONL}\\n")

successes, failures = [], []
matrix_start = time.perf_counter()

for idx, (tag, flags) in enumerate(configs_to_run, start=1):
    print(
        f"[{idx}/{len(configs_to_run)}] {tag:12s}  "
        f"dp={flags['dp']}  bie={flags['bie']}  smpc={flags['smpc']}"
    )
    cfg = build_config(tag, flags)
    image_ppts, embedding_ppts = build_ppts(flags)
    run_start = time.perf_counter()
    try:
        result = run_single_config(cfg, image_ppts, embedding_ppts)
        elapsed = time.perf_counter() - run_start
        append_run_result(result, RUNS_JSONL)
        successes.append(tag)
        yeom_acc = (
            f"{result.privacy.yeom.attack_accuracy:.3f}"
            if result.privacy.yeom else "—"
        )
        shokri_acc = (
            f"{result.privacy.shokri.attack_accuracy:.3f}"
            if result.privacy.shokri else "—"
        )
        print(
            f"   OK test_acc={result.utility.test_accuracy:.3f}  "
            f"f1={result.utility.f1:.3f}  "
            f"yeom={yeom_acc}  "
            f"shokri={shokri_acc}  "
            f"({_fmt_duration(elapsed)})\\n"
        )
    except Exception as e:
        elapsed = time.perf_counter() - run_start
        failures.append((tag, repr(e)))
        print(f"   !! FAILED after {_fmt_duration(elapsed)}: {e}\\n")

matrix_elapsed = time.perf_counter() - matrix_start
print(f"\\nMatrix complete in {_fmt_duration(matrix_elapsed)}.")
print(f"  OK Successes ({len(successes)}): {successes}")
if failures:
    print(f"  !! Failures ({len(failures)}):")
    for tag, err in failures:
        print(f"    - {tag}: {err}")
"""


CELL_10_DATAFRAME = """\
# Cell 10 — View results as a pandas DataFrame
import json
from pathlib import Path

import pandas as pd

RUNS_JSONL = Path("/content/Privacy_Preserving_ML/results/runs.jsonl")

if not RUNS_JSONL.exists():
    raise FileNotFoundError(
        f"No results at {RUNS_JSONL}. Run Cell 9 first."
    )

records = []
with RUNS_JSONL.open() as f:
    for raw_line in f:
        stripped = raw_line.strip()
        if stripped:
            records.append(json.loads(stripped))

if not records:
    raise RuntimeError(f"{RUNS_JSONL} is empty.")


def flatten(rec: dict) -> dict:
    config = rec["config"]
    util = rec["utility"]
    eff = rec["efficiency"]
    yeom = rec["privacy"].get("yeom")
    shokri = rec["privacy"].get("shokri")
    return {
        "tag": rec["tag"],
        "dp": config["dp_enabled"],
        "dp_eps": config["dp_epsilon"] if config["dp_enabled"] else None,
        "bie": config["bie_enabled"],
        "smpc": config["smpc_enabled"],
        "test_acc": util["test_accuracy"],
        "f1": util["f1"],
        "ece": util["ece"],
        "yeom_acc": yeom["attack_accuracy"] if yeom else None,
        "yeom_auc": yeom["attack_auc"] if yeom else None,
        "shokri_acc": shokri["attack_accuracy"] if shokri else None,
        "shokri_auc": shokri["attack_auc"] if shokri else None,
        "train_s": eff["train_latency_seconds"],
        "inf_ms": eff["inference_latency_ms_per_query"],
        "mem_mb": eff["memory_peak_mb"],
        "timestamp": rec["timestamp"],
    }


df = pd.DataFrame([flatten(r) for r in records])

# If the notebook is rerun, keep only the latest record per tag
df = (
    df.sort_values("timestamp")
      .drop_duplicates("tag", keep="last")
      .reset_index(drop=True)
)

print(f"{len(df)} run record(s) loaded from {RUNS_JSONL.name}\\n")

pd.set_option("display.float_format", lambda x: f"{x:.3f}")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print(df.to_string(index=False))

CSV_PATH = RUNS_JSONL.parent / "runs.csv"
df.to_csv(CSV_PATH, index=False)
print(f"\\nAlso saved to {CSV_PATH} (ready for the paper's Results table).")
"""


CELL_11_DOWNLOAD = """\
# Cell 11 — Download results to your local machine
from google.colab import files

RESULTS_DIR = "/content/Privacy_Preserving_ML/results"
print("Downloading runs.jsonl and runs.csv...")
files.download(f"{RESULTS_DIR}/runs.jsonl")
files.download(f"{RESULTS_DIR}/runs.csv")

print("\\nFiles should appear in your browser's Downloads folder.")
print("\\nTo commit them to the repo:")
print("  cd /path/to/local/Privacy_Preserving_ML")
print("  mv ~/Downloads/runs.jsonl ~/Downloads/runs.csv results/")
print("  git add -f results/runs.jsonl results/runs.csv  # -f: results/ is gitignored")
print("  git commit -m 'Results from Colab run YYYY-MM-DD'")
print("  git push origin main")
"""


CELL_12_NEXT_STEPS = """\
---

## Next Steps

1. **Commit the results to the repo** (teammates need to see them):
   ```bash
   cd /path/to/local/Privacy_Preserving_ML
   mv ~/Downloads/runs.jsonl ~/Downloads/runs.csv results/
   git add -f results/runs.jsonl results/runs.csv
   git commit -m "Results from Colab run YYYY-MM-DD"
   git push origin main
   ```

2. **Interpret the numbers**:
   - *Privacy*: how close is `yeom_acc` / `shokri_acc` to 0.50? Closer = stronger defense.
   - *Utility*: how much does `test_acc` drop from baseline? Target per spec §8: <5%.
   - *Trade-off*: plot privacy vs utility per config.

3. **Per-paper lit-review paragraphs** in `conference_101719.tex`:
   - Eray — Paper 1 (Kiya et al. 2023, Block-wise ViT)
   - Onur — Paper 2 (Ziller et al. 2021, DP for medical imaging)
   - Egehan — Paper 3 (Jarin & Eshete 2021, PRICURE)

4. **Slide amendment**: update presentation slide 3 from "standard pretrained CNN backbone"
   to "custom 3-conv CNN trained from scratch" (per spec §2 deviation note).

5. **Methods + Results sections**: drop the `runs.csv` numbers into the paper.
"""


def build_notebook() -> dict:
    return {
        "cells": [
            md_cell(CELL_1_MARKDOWN),
            code_cell(CELL_2_GPU_CHECK),
            code_cell(CELL_3_CLONE),
            code_cell(CELL_4_INSTALL),
            code_cell(CELL_5_PYTEST),
            code_cell(CELL_6_DOWNLOAD),
            code_cell(CELL_7_VERIFY),
            md_cell(CELL_8_SEPARATOR),
            code_cell(CELL_9_MATRIX),
            code_cell(CELL_10_DATAFRAME),
            code_cell(CELL_11_DOWNLOAD),
            md_cell(CELL_12_NEXT_STEPS),
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
            "colab": {"provenance": [], "toc_visible": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    nb = build_notebook()
    OUTPUT_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"Wrote {OUTPUT_PATH} ({OUTPUT_PATH.stat().st_size} bytes, "
          f"{len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
