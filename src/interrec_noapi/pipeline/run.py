from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_CONFIGS = {
    "lastfm": PROJECT_ROOT / "configs/runs/lastfm_full_seed42.yaml",
    "movielens": PROJECT_ROOT / "configs/runs/movielens_full_seed42.yaml",
    "book": PROJECT_ROOT / "configs/runs/book_full_seed42.yaml",
}


def run_dataset(dataset: str, config: str | None = None, extra_args: Sequence[str] = ()) -> int:
    dataset_key = dataset.lower()
    cfg = Path(config) if config else DATASET_CONFIGS.get(dataset_key)
    if cfg is None:
        raise ValueError(f"unknown dataset: {dataset}")
    if not cfg.is_absolute():
        cfg = PROJECT_ROOT / cfg
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts/run_main_experiment.py"), "--config", str(cfg), *extra_args]
    return subprocess.call(cmd, cwd=str(PROJECT_ROOT))
