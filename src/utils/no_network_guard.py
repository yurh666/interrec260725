from __future__ import annotations

import builtins
import json
import os
import socket
import time
import urllib.request
from pathlib import Path
from types import ModuleType
from typing import Any

_INSTALLED = False
_ORIGINAL_IMPORT = builtins.__import__
_ORIGINAL_SOCKET_CONNECT = socket.socket.connect
_ORIGINAL_CREATE_CONNECTION = socket.create_connection
_ORIGINAL_URLOPEN = urllib.request.urlopen

_BLOCKED_MODULES = ("openai", "anthropic", "requests", "httpx", "aiohttp")


def _log_event(kind: str, target: str) -> None:
    path = os.environ.get("INTERREC_NETWORK_GUARD_LOG")
    if not path:
        return
    row = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "kind": kind,
        "target": target,
        "blocked": True,
    }
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    except OSError:
        pass


def _guard_error(kind: str, target: str) -> RuntimeError:
    _log_event(kind, target)
    return RuntimeError(f"Network guard blocked {kind}: {target}")


def _guarded_import(name: str, globals: Any = None, locals: Any = None, fromlist: Any = (), level: int = 0) -> ModuleType:
    root = str(name).split(".", 1)[0]
    if root in _BLOCKED_MODULES:
        raise _guard_error("import", str(name))
    return _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)


def _guarded_connect(self: socket.socket, address: Any) -> Any:
    raise _guard_error("socket.connect", repr(address))


def _guarded_create_connection(address: Any, *args: Any, **kwargs: Any) -> Any:
    raise _guard_error("socket.create_connection", repr(address))


def _guarded_urlopen(url: Any, *args: Any, **kwargs: Any) -> Any:
    raise _guard_error("urllib.request.urlopen", repr(url))


def install_no_network_guard() -> dict[str, Any]:
    global _INSTALLED
    if _INSTALLED:
        return status()
    builtins.__import__ = _guarded_import
    socket.socket.connect = _guarded_connect
    socket.create_connection = _guarded_create_connection
    urllib.request.urlopen = _guarded_urlopen
    _INSTALLED = True
    return status()


def status() -> dict[str, Any]:
    return {
        "network_guard_active": bool(_INSTALLED),
        "expected_api_calls": 0,
        "blocked_imports": list(_BLOCKED_MODULES),
        "blocked_calls": [
            "socket.socket.connect",
            "socket.create_connection",
            "urllib.request.urlopen",
        ],
        "log_path": os.environ.get("INTERREC_NETWORK_GUARD_LOG"),
    }
