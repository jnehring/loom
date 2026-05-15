"""Persistence for batch metadata under ~/.loom/batches/."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..core.models import BatchMetadata

STORAGE_DIR = Path.home() / ".loom" / "batches"


def _ensure_dir() -> None:
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)


def _path_for(meta: BatchMetadata) -> Path:
    safe_id = meta.batch_id.replace("/", "_")
    return STORAGE_DIR / f"{meta.provider}_{safe_id}.json"


def save_batch(meta: BatchMetadata) -> Path:
    _ensure_dir()
    p = _path_for(meta)
    p.write_text(meta.model_dump_json(indent=2))
    return p


def load_batch(batch_id: str) -> BatchMetadata:
    _ensure_dir()
    safe_id = batch_id.replace("/", "_")
    for f in STORAGE_DIR.glob(f"*_{safe_id}.json"):
        return BatchMetadata.model_validate_json(f.read_text())
    raise FileNotFoundError(f"No stored batch found for id={batch_id}")


def list_batches() -> list[BatchMetadata]:
    _ensure_dir()
    out: list[BatchMetadata] = []
    for f in sorted(STORAGE_DIR.glob("*.json")):
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
    for f in STORAGE_DIR.glob(f"*_{safe_id}.json"):
        f.unlink()
        removed = True
    return removed
