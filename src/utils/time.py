from __future__ import annotations

from datetime import datetime


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def timestamp_compact() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def make_run_id(dataset: str, method: str, seed: int, *, extra_suffix: str = "") -> str:
    base = f"{timestamp_compact()}_{dataset}_{method}_seed{seed}"
    return f"{base}{extra_suffix}" if extra_suffix else base
