#!/usr/bin/env python3
"""Convert hetrec2011-lastfm-2k data to InterRec interactions.csv / items.csv.

The hetrec2011-lastfm-2k dataset uses ARTIST as item granularity
(not individual tracks). Each user-artist pair has a `weight` = play count.

Input files (all inside the extracted zip directory):
    user_artists.dat   — userID, artistID, weight (play count)
    artists.dat        — id, name, url, pictureURL
    user_taggedartists-timestamps.dat — userID, artistID, tagID, timestamp
    tags.dat           — tagID, tagValue

Output (in --out directory):
    interactions.csv   — user_id, item_id, timestamp, play_count
    items.csv          — item_id, title, artist_name, category, tags, description

Usage::

    python scripts/preprocess_lastfm.py \\
        --raw data/raw/hetrec2011-lastfm-2k \\
        --out data/raw
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw/hetrec2011-lastfm-2k")
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()

    raw = Path(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── Load source files ──────────────────────────────────────────────────
    print(f"[lastfm] Loading from {raw}")
    ua = pd.read_csv(raw / "user_artists.dat", sep="\t", encoding="latin-1")
    artists = pd.read_csv(raw / "artists.dat", sep="\t", usecols=["id", "name"], encoding="latin-1")
    artists.columns = ["artistID", "artist_name"]

    tags = pd.read_csv(raw / "tags.dat", sep="\t", encoding="latin-1")
    tags.columns = ["tagID", "tagValue"]
    tag_map = dict(zip(tags["tagID"], tags["tagValue"]))

    tagged = pd.read_csv(
        raw / "user_taggedartists-timestamps.dat",
        sep="\t",
        usecols=["userID", "artistID", "tagID", "timestamp"],
        encoding="latin-1",
    )

    # ── Build per-artist tag list ──────────────────────────────────────────
    artist_tags = (
        tagged.groupby("artistID")["tagID"]
        .apply(lambda x: " ".join(tag_map.get(t, str(t)) for t in x.unique()[:20]))
        .reset_index()
    )
    artist_tags.columns = ["artistID", "tags"]

    # ── Build items.csv ────────────────────────────────────────────────────
    items = artists.merge(artist_tags, on="artistID", how="left").fillna("")
    items["item_id"] = items["artistID"].astype(str)
    items["title"] = items["artist_name"]
    items["track_name"] = ""
    items["category"] = ""
    items["description"] = items["artist_name"] + ". Tags: " + items["tags"]
    items = items[["item_id", "title", "artist_name", "track_name", "category", "tags", "description"]]
    items.to_csv(out / "items.csv", index=False)
    print(f"[lastfm] Saved items: {len(items)} artists -> {out / 'items.csv'}")

    # ── Build interactions.csv ─────────────────────────────────────────────
    # Use play_count as a proxy timestamp rank: users who listen more have later
    # "effective" positions. Sort by play count descending so most-played =
    # end of sequence (recent preference signal).
    interactions = ua.rename(columns={"userID": "user_id", "artistID": "item_id", "weight": "play_count"})
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["item_id"] = interactions["item_id"].astype(str)
    # Assign a within-user timestamp rank based on play_count ascending
    # (lower play_count → earlier discovery; higher → more recent / stronger)
    interactions = interactions.sort_values(["user_id", "play_count"], ascending=[True, True])
    interactions["timestamp"] = interactions.groupby("user_id").cumcount()

    # Merge real tag-based timestamps if available (use min tag timestamp per user-artist)
    min_ts = (
        tagged.groupby(["userID", "artistID"])["timestamp"]
        .min()
        .reset_index()
    )
    min_ts.columns = ["user_id", "item_id", "real_timestamp"]
    min_ts["user_id"] = min_ts["user_id"].astype(str)
    min_ts["item_id"] = min_ts["item_id"].astype(str)
    interactions = interactions.merge(min_ts, on=["user_id", "item_id"], how="left")
    # Use real_timestamp when available, else fall back to play_count rank
    interactions["timestamp"] = interactions["real_timestamp"].fillna(
        interactions["timestamp"].astype(float)
    ).astype(float)
    interactions = interactions[["user_id", "item_id", "timestamp", "play_count"]]
    interactions = interactions.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    interactions.to_csv(out / "interactions.csv", index=False)
    print(f"[lastfm] Saved interactions: {len(interactions)} rows -> {out / 'interactions.csv'}")
    print(f"[lastfm] Users: {interactions['user_id'].nunique()}, Items: {interactions['item_id'].nunique()}")


if __name__ == "__main__":
    main()
