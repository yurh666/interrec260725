from __future__ import annotations

import importlib
import traceback
import urllib.request
from typing import Any
from unittest.mock import patch


_PATCHES: list[Any] = []
_BLOCKED_CALLS: list[dict[str, Any]] = []
_ACTIVE = False


def _blocker(kind: str):
    def blocked(*args: Any, **kwargs: Any) -> None:
        _BLOCKED_CALLS.append(
            {
                "kind": kind,
                "args_repr": repr(args[:2]),
                "kwargs_keys": sorted(str(k) for k in kwargs),
                "stack": traceback.format_stack(limit=24),
            }
        )
        raise RuntimeError(f"NETWORK_CALL_BLOCKED:{kind}")

    return blocked


class _BlockedOpenAI:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _blocker("openai.OpenAI")(*args, **kwargs)


class _BlockedAnthropic:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        _blocker("anthropic.Anthropic")(*args, **kwargs)


def _optional_patch(target: str, replacement: Any) -> Any | None:
    try:
        module_name, _attr = target.rsplit(".", 1)
        importlib.import_module(module_name)
        return patch(target, replacement)
    except Exception:
        return None


def install_network_guard(enabled: bool = True) -> dict[str, Any]:
    """Install process-local network blockers for API-free evaluation runs."""

    global _ACTIVE
    if not enabled:
        return snapshot_network_guard()
    if _ACTIVE:
        return snapshot_network_guard()

    targets: tuple[tuple[str, Any], ...] = (
        ("openai.OpenAI", _BlockedOpenAI),
        ("anthropic.Anthropic", _BlockedAnthropic),
        ("requests.request", _blocker("requests.request")),
        ("requests.get", _blocker("requests.get")),
        ("requests.post", _blocker("requests.post")),
        ("httpx.request", _blocker("httpx.request")),
        ("httpx.get", _blocker("httpx.get")),
        ("httpx.post", _blocker("httpx.post")),
        ("httpx.Client", _blocker("httpx.Client")),
        ("httpx.AsyncClient", _blocker("httpx.AsyncClient")),
        ("aiohttp.ClientSession", _blocker("aiohttp.ClientSession")),
        ("urllib.request.urlopen", _blocker("urllib.request.urlopen")),
    )
    for target, replacement in targets:
        patched = _optional_patch(target, replacement)
        if patched is not None:
            _PATCHES.append(patched)
    for patched in _PATCHES:
        patched.start()
    # urllib is in stdlib and always available; ensure it is blocked even if importlib patch was skipped.
    if not any("urllib.request.urlopen" in str(p) for p in _PATCHES):
        p = patch.object(urllib.request, "urlopen", _blocker("urllib.request.urlopen"))
        _PATCHES.append(p)
        p.start()
    _ACTIVE = True
    return snapshot_network_guard()


def snapshot_network_guard() -> dict[str, Any]:
    return {
        "network_guard_active": bool(_ACTIVE),
        "api_calls_detected": int(len(_BLOCKED_CALLS)),
        "blocked_call_count": int(len(_BLOCKED_CALLS)),
        "blocked_calls": list(_BLOCKED_CALLS),
    }


def reset_network_guard_counts() -> None:
    _BLOCKED_CALLS.clear()
