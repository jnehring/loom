"""On-disk response cache for sync (non-batch) generation.

Keyed by sha256(provider | model | prompt). Stored as one small JSON file per
entry under ``~/.loom/cache/``. The cache is opt-in via ``use_cache=True`` in
``orchestrator.generate_sync`` (the CLI exposes ``--no-cache`` to disable).

The cache is intentionally simple — no eviction, no TTL. Users can clear it by
removing ``~/.loom/cache/`` (or specific files).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


CACHE_DIR = Path.home() / ".loom" / "cache"


def _ensure_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _key(provider: str, model: str, prompt: str) -> str:
    h = hashlib.sha256()
    h.update(provider.encode("utf-8"))
    h.update(b"\x00")
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()


def _path(provider: str, model: str, prompt: str) -> Path:
    return CACHE_DIR / f"{_key(provider, model, prompt)}.json"


def get(provider: str, model: str, prompt: str) -> Optional[str]:
    p = _path(provider, model, prompt)
    if not p.exists():
        return None
    try:
        obj = json.loads(p.read_text())
        resp = obj.get("response")
        return resp if isinstance(resp, str) else None
    except Exception:  # noqa: BLE001
        return None


def set(provider: str, model: str, prompt: str, response: str) -> None:  # noqa: A001
    _ensure_dir()
    obj = {
        "provider": provider,
        "model": model,
        "response": response,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _path(provider, model, prompt).write_text(
        json.dumps(obj, ensure_ascii=False)
    )


def clear() -> int:
    """Delete every cached entry. Returns the number of files removed."""
    if not CACHE_DIR.exists():
        return 0
    n = 0
    for f in CACHE_DIR.glob("*.json"):
        try:
            f.unlink()
            n += 1
        except OSError:
            pass
    return n
