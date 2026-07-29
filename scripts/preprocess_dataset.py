#!/usr/bin/env python3
"""Preprocess raw interaction and item data into the processed/ directory.

Usage
-----
From the interrec/ project root::

    python scripts/preprocess_dataset.py --config configs/default.yaml

The script reads raw files specified in the config, filters, splits and writes:
    data/processed/interactions.csv
    data/processed/items.csv
    data/processed/user_splits.json
    data/processed/sessions.json

The script never touches future_test data for any system module — all splitting
is done by split_user_history.py and only stored to disk.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocess import preprocess_dataset
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess InterRec dataset")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"[preprocess] Dataset: {cfg['dataset']['name']}")
    print(f"[preprocess] Raw interactions: {cfg['dataset']['raw_interactions_path']}")

    result = preprocess_dataset(cfg)
    print("[preprocess] Done.")
    for k, v in result.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
