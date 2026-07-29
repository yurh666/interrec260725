#!/usr/bin/env python3
"""Shared utilities for ask event extraction from existing logs."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]

PRIMARY_LOGS = {
    "lastfm": [
        ROOT / "experiments/final_results_seed42/lastfm/full_log.jsonl",
    ],
    "movielens": [
        ROOT / "experiments/final_results_seed42/movielens/full_log.jsonl",
    ],
    "book": [
        ROOT / "experiments/final_results_seed42/book/full_log.jsonl",
    ],
}

METADATA_PATHS = {
    "lastfm": ROOT / "data/processed/items.csv",
    "movielens": ROOT / "data/processed_pack_20260523/processed/movielens/items.csv",
    "book": ROOT / "data/processed_pack_20260523/processed/amazon_book/enriched_items.csv",
}

LEAKAGE_KEYS = frozenset({
    "theta_star", "theta_star_items", "selected_option_id", "selected_vector_index",
    "user_choice", "ground_truth", "target_item", "future_ranking",
})


def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def parse_feature_signature(sig: list[Any] | None) -> tuple[str, str]:
    if not sig:
        return "", ""
    parts = [str(x) for x in sig]
    cluster_id = parts[0][3:] if parts and parts[0].startswith("cb_") else (parts[0] if parts else "")
    role = parts[1] if len(parts) > 1 else ""
    return cluster_id, role


def extract_names_from_text(text: str) -> list[str]:
    if "especially items like" in text:
        tail = text.split("especially items like", 1)[1]
        names = re.split(r",|\band\b", tail)
        return [n.strip().rstrip(".") for n in names if n.strip()][:5]
    return []


def option_vector_hashes(hyps: list[dict[str, Any]]) -> list[str]:
    out = []
    for h in hyps:
        payload = json.dumps(
            {
                "id": h.get("hypothesis_id"),
                "reps": h.get("representative_items") or h.get("parsed_representative_items"),
                "sig": h.get("feature_signature"),
            },
            sort_keys=True,
        )
        out.append(hashlib.sha256(payload.encode()).hexdigest()[:16])
    return out


def payload_hash(dataset: str, episode_id: str, turn: int, hyps: list[dict[str, Any]]) -> str:
    payload = {
        "dataset": dataset,
        "episode_id": episode_id,
        "turn": turn,
        "options": [
            {
                "option_id": h.get("hypothesis_id"),
                "representative_item_ids": h.get("representative_items") or h.get("parsed_representative_items"),
                "feature_signature": h.get("feature_signature"),
            }
            for h in hyps
        ],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def load_item_name_map(dataset: str) -> dict[str, str]:
    path = METADATA_PATHS.get(dataset)
    if not path or not path.exists():
        return {}
    mapping: dict[str, str] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iid = str(row.get("item_id") or "").strip()
            if not iid:
                continue
            if dataset == "lastfm":
                name = (row.get("title") or row.get("artist_name") or row.get("track_name") or "").strip()
            elif dataset == "book":
                name = (row.get("title") or "").strip()
                if row.get("author"):
                    name = f"{name} ({row['author']})" if name else str(row["author"])
            else:
                name = (row.get("title") or "").strip()
            if name:
                mapping[iid] = name
    return mapping


def resolve_item_names(item_ids: list[str], name_map: dict[str, str]) -> list[str]:
    names = []
    for iid in item_ids:
        names.append(name_map.get(str(iid), str(iid)))
    return names
