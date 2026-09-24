"""On-disk response cache for sync and batch generation.

Keyed by sha256(provider | model | prompt [| settings]). The settings part is the
canonical JSON of the non-default :class:`~loom.core.models.GenerationParams`
and is left out when every setting is at its default, so keys created before
generation settings existed stay valid. Stored as one small JSON file per
entry under the configured cache directory (default ``~/.loom/cache/``).

The cache is intentionally simple — no eviction, no TTL. Users can clear it via
``ResponseCache.clear()``, ``loom cache clear``, or by deleting the directory.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from .paths import default_cache_dir

if TYPE_CHECKING:
    from ..core.models import GenerationParams

PathLike = Union[str, Path]


def _token(params: "Optional[GenerationParams]") -> str:
    return params.cache_token() if params is not None else ""


class ResponseCache:
    """On-disk cache of LLM responses keyed by (provider, model, prompt).

    Pass ``enabled=False`` for a no-op instance that never reads or writes.
    Pass ``cache_dir`` to override the location; otherwise the directory is
    resolved via :func:`default_cache_dir`.
    """

    def __init__(
        self,
        cache_dir: Optional[PathLike] = None,
        *,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._dir = Path(cache_dir).expanduser() if cache_dir else default_cache_dir()

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def dir(self) -> Path:
        return self._dir

    def _ensure_dir(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(provider: str, model: str, prompt: str, params: "Optional[GenerationParams]" = None) -> str:
        h = hashlib.sha256()
        h.update(provider.encode("utf-8"))
        h.update(b"\x00")
        h.update(model.encode("utf-8"))
        h.update(b"\x00")
        h.update(prompt.encode("utf-8"))
        token = _token(params)
        if token:
            h.update(b"\x00")
            h.update(token.encode("utf-8"))
        return h.hexdigest()

    def _path(self, provider: str, model: str, prompt: str, params: "Optional[GenerationParams]" = None) -> Path:
        return self._dir / f"{self.key(provider, model, prompt, params)}.json"

    def get(self, provider: str, model: str, prompt: str, params: "Optional[GenerationParams]" = None) -> Optional[str]:
        if not self._enabled:
            return None
        p = self._path(provider, model, prompt, params)
        if not p.exists():
            return None
        try:
            obj = json.loads(p.read_text())
            resp = obj.get("response")
            return resp if isinstance(resp, str) else None
        except Exception:  # noqa: BLE001
            return None

    def set(  # noqa: A001
        self, provider: str, model: str, prompt: str, response: str, params: "Optional[GenerationParams]" = None
    ) -> None:
        if not self._enabled:
            return
        self._ensure_dir()
        obj = {
            "provider": provider,
            "model": model,
            "response": response,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        token = _token(params)
        if token:
            obj["params"] = json.loads(token)
        self._path(provider, model, prompt, params).write_text(
            json.dumps(obj, ensure_ascii=False)
        )

    def count(self) -> int:
        """Return the number of cached entry files (0 if the directory is missing)."""
        if not self._dir.exists():
            return 0
        return sum(1 for _ in self._dir.glob("*.json"))

    def clear(self) -> int:
        """Delete every cached entry. Returns the number of files removed."""
        if not self._dir.exists():
            return 0
        n = 0
        for f in self._dir.glob("*.json"):
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
        return n


# ---------- Module-level delegates (default cache under ~/.loom/cache/) ----------

_default: Optional[ResponseCache] = None


def _default_cache() -> ResponseCache:
    global _default
    if _default is None:
        _default = ResponseCache()
    return _default


def reset_default_cache() -> None:
    """Drop the cached default instance so the next call re-resolves paths/env."""
    global _default
    _default = None


def get_cache(
    cache_dir: Optional[PathLike] = None,
    *,
    enabled: bool = True,
) -> ResponseCache:
    """Return a :class:`ResponseCache`.

    If ``cache_dir`` is ``None`` and ``enabled`` is True, returns the shared
    default instance. Otherwise constructs a fresh instance.
    """
    if cache_dir is None and enabled:
        return _default_cache()
    return ResponseCache(cache_dir=cache_dir, enabled=enabled)


# Backwards-compatible module attributes / functions used by older callers.
CACHE_DIR = default_cache_dir()  # snapshot at import; prefer ResponseCache.dir


def get(provider: str, model: str, prompt: str, params: "Optional[GenerationParams]" = None) -> Optional[str]:
    return _default_cache().get(provider, model, prompt, params)


def set(  # noqa: A001
    provider: str, model: str, prompt: str, response: str, params: "Optional[GenerationParams]" = None
) -> None:
    _default_cache().set(provider, model, prompt, response, params)


def clear() -> int:
    """Delete every cached entry under the default cache directory."""
    return _default_cache().clear()
