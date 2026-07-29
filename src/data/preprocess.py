from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.data.split_user_history import split_grouped_histories
from src.utils.io import ensure_dir, write_json


INTERACTION_ALIASES = {
    "user": "user_id",
    "uid": "user_id",
    "artist": "item_id",
    "track": "item_id",
    "item": "item_id",
    "time": "timestamp",
    "date": "timestamp",
}


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    lowered = {c.lower(): c for c in df.columns}
    for alias, canonical in INTERACTION_ALIASES.items():
        if alias in lowered and canonical not in df.columns:
            rename[lowered[alias]] = canonical
    return df.rename(columns=rename)


def infer_item_id(df: pd.DataFrame) -> pd.Series:
    if "item_id" in df.columns:
        return df["item_id"].astype(str)
    if {"artist_name", "track_name"}.issubset(df.columns):
        return (df["artist_name"].fillna("") + "::" + df["track_name"].fillna("")).astype(str)
    if "artist_name" in df.columns:
        return df["artist_name"].astype(str)
    raise ValueError("Interactions must contain item_id, artist_name, or artist_name + track_name.")


def load_interactions(path: str | Path) -> pd.DataFrame:
    df = normalize_columns(pd.read_csv(path))
    if "user_id" not in df.columns:
        raise ValueError("Interactions must contain user_id.")
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = infer_item_id(df)
    if "timestamp" not in df.columns:
        df["timestamp"] = df.groupby("user_id").cumcount()
    return df


def load_items(path: str | Path | None, interactions: pd.DataFrame) -> pd.DataFrame:
    if path and Path(path).exists():
        items = normalize_columns(pd.read_csv(path))
        if "item_id" not in items.columns:
            items["item_id"] = infer_item_id(items)
    else:
        cols = [c for c in ["item_id", "title", "artist_name", "track_name", "genre", "category", "tags", "description"] if c in interactions.columns]
        items = interactions[cols].drop_duplicates("item_id")
    items["item_id"] = items["item_id"].astype(str)
    for col in ["title", "description", "category", "tags", "artist_name", "track_name", "genre"]:
        if col not in items.columns:
            items[col] = ""
    return items.drop_duplicates("item_id")


def filter_interactions(df: pd.DataFrame, min_user_interactions: int, min_item_interactions: int) -> pd.DataFrame:
    filtered = df.copy()
    changed = True
    while changed:
        before = len(filtered)
        user_counts = filtered["user_id"].value_counts()
        filtered = filtered[filtered["user_id"].isin(user_counts[user_counts >= min_user_interactions].index)]
        item_counts = filtered["item_id"].value_counts()
        filtered = filtered[filtered["item_id"].isin(item_counts[item_counts >= min_item_interactions].index)]
        changed = len(filtered) != before
    return filtered.copy()


def preprocess_dataset(config: dict[str, Any]) -> dict[str, Any]:
    ds_cfg = config["dataset"]
    processed_dir = ensure_dir(ds_cfg.get("processed_dir", "data/processed"))
    interactions = load_interactions(ds_cfg["raw_interactions_path"])
    interactions = filter_interactions(
        interactions,
        int(ds_cfg.get("min_user_interactions", 20)),
        int(ds_cfg.get("min_item_interactions", 5)),
    )
    interactions = interactions.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    items = load_items(ds_cfg.get("raw_items_path"), interactions)
    items = items[items["item_id"].isin(interactions["item_id"].unique())].copy()

    histories = {
        user_id: group.sort_values("timestamp")["item_id"].astype(str).tolist()
        for user_id, group in interactions.groupby("user_id", sort=False)
    }
    splits = split_grouped_histories(histories, ds_cfg)
    splits = [
        s for s in splits
        if len(s["observed_history"]) >= int(ds_cfg.get("min_observed_interactions", 10))
        and (len(s["future_train"]) + len(s["future_valid"]) + len(s["future_test"])) >= int(ds_cfg.get("min_future_interactions", 5))
    ]
    keep_users = {s["user_id"] for s in splits}
    interactions = interactions[interactions["user_id"].isin(keep_users)].copy()

    sessions = [
        {
            "episode_id": f"{s['user_id']}_session",
            "user_id": s["user_id"],
            "observed_history": s["observed_history"],
            "future_train": s["future_train"],
            "future_valid": s["future_valid"],
            "future_test": s["future_test"],
        }
        for s in splits
    ]

    interactions.to_csv(processed_dir / "interactions.csv", index=False)
    items.to_csv(processed_dir / "items.csv", index=False)
    write_json(splits, processed_dir / "user_splits.json")
    write_json(sessions, processed_dir / "sessions.json")
    return {
        "interactions": str(processed_dir / "interactions.csv"),
        "items": str(processed_dir / "items.csv"),
        "user_splits": str(processed_dir / "user_splits.json"),
        "sessions": str(processed_dir / "sessions.json"),
        "n_interactions": int(len(interactions)),
        "n_items": int(items["item_id"].nunique()),
        "n_users": int(len(splits)),
    }
