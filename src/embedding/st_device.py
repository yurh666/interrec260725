"""SentenceTransformer device resolution & CUDA probes for reproducible embedding builds."""

from __future__ import annotations

import sys


def probe_torch_cuda() -> tuple[bool, str | None]:
    """Return (cuda_available, primary_gpu_name or None)."""
    try:
        import torch

        ok = torch.cuda.is_available()
        name = torch.cuda.get_device_name(0) if ok else None
        return bool(ok), name
    except ImportError:
        return False, None


def resolve_sentence_transformer_device_string(preference: str) -> str:
    """Resolve config ``embedding.device`` to a string usable by SentenceTransformer(..., device=...).

    Parameters
    ----------
    preference
        ``auto`` | ``cuda`` | ``cpu``

    Raises
    ------
    SystemExit
        If preference is ``cuda`` but CUDA is not available (exit code 3).
    """
    pref = (preference or "auto").strip().lower()
    if pref not in {"auto", "cuda", "cpu"}:
        print(f"[embed][device] ERROR: invalid embedding.device={preference!r} (allowed: auto, cuda, cpu)", file=sys.stderr)
        sys.exit(2)

    try:
        import torch
    except ImportError:
        torch = None

    cuda_ok = False
    if torch is not None:
        cuda_ok = bool(torch.cuda.is_available())

    if pref == "cpu":
        return "cpu"
    if pref == "cuda":
        if not cuda_ok:
            print(
                "[embed][device] ERROR: embedding.device=cuda but torch.cuda.is_available() is False. "
                "Fix drivers / CUDA runtime or use device: auto/cpu.",
                file=sys.stderr,
            )
            sys.exit(3)
        return "cuda"
    # auto
    return "cuda" if cuda_ok else "cpu"
