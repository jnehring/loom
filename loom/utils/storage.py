"""Persistence for batch metadata under ``<loom_home>/batches/`` (``$LOOM_HOME`` or ``~/.loom``)."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union

from ..core.models import BatchMetadata
from .paths import default_batches_dir, default_inputs_dir

PathLike = Union[str, Path]

# Optional overrides (rebindable in tests via monkeypatch). ``None`` means: resolve ``$LOOM_HOME`` at call time, so a
# caller that sets ``LOOM_HOME`` after importing loom still gets its own directory.
STORAGE_DIR: Optional[Path] = None
INPUTS_DIR: Optional[Path] = None


def storage_dir() -> Path:
    return STORAGE_DIR if STORAGE_DIR is not None else default_batches_dir()


def inputs_dir() -> Path:
    return INPUTS_DIR if INPUTS_DIR is not None else default_inputs_dir()


def _ensure_dir(path: Optional[Path] = None) -> Path:
    d = path if path is not None else storage_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path_for(meta: BatchMetadata) -> Path:
    safe_id = meta.batch_id.replace("/", "_")
    return storage_dir() / f"{meta.provider}_{safe_id}.json"


def save_batch(meta: BatchMetadata) -> Path:
    _ensure_dir()
    p = _path_for(meta)
    p.write_text(meta.model_dump_json(indent=2))
    return p


def load_batch(batch_id: str) -> BatchMetadata:
    _ensure_dir()
    safe_id = batch_id.replace("/", "_")
    for f in storage_dir().glob(f"*_{safe_id}.json"):
        return BatchMetadata.model_validate_json(f.read_text())
    raise FileNotFoundError(f"No stored batch found for id={batch_id}")


def list_batches() -> list[BatchMetadata]:
    _ensure_dir()
    out: list[BatchMetadata] = []
    for f in sorted(storage_dir().glob("*.json")):
        try:
            out.append(BatchMetadata.model_validate_json(f.read_text()))
        except Exception:
            continue

    def _sort_key(m: BatchMetadata) -> datetime:
        dt = m.created_at
        return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt

    out.sort(key=_sort_key, reverse=True)
    return out


def delete_batch(batch_id: str) -> bool:
    _ensure_dir()
    safe_id = batch_id.replace("/", "_")
    removed = False
    for f in storage_dir().glob(f"*_{safe_id}.json"):
        f.unlink()
        removed = True
    return removed


def save_memory_input(provider: str, batch_id: str, prompts: list[dict]) -> Path:
    """Persist an in-memory prompt list so batch fetch can re-load it later.

    Writes a JSON list of ``{id, prompt}`` objects under
    ``<loom_home>/inputs/<provider>_<batch_id>.json``.
    """
    import json

    d = _ensure_dir(inputs_dir())
    safe_id = batch_id.replace("/", "_")
    path = d / f"{provider}_{safe_id}.json"
    path.write_text(json.dumps(prompts, indent=2, ensure_ascii=False))
    return path


def delete_memory_input(provider: str, batch_id: str) -> bool:
    """Delete a previously saved in-memory input file. Returns True if removed."""
    safe_id = batch_id.replace("/", "_")
    path = inputs_dir() / f"{provider}_{safe_id}.json"
    if path.exists():
        path.unlink()
        return True
    return False
