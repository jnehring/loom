from __future__ import annotations

from pathlib import Path
from ..core.models import BatchMetadata


LOOM_DIR = Path.home() / ".loom"
BATCHES_DIR = LOOM_DIR / "batches"


def ensure_dirs() -> None:
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)


def _path_for(provider: str, batch_id: str) -> Path:
    return BATCHES_DIR / f"{provider}_{batch_id}.json"


def save(meta: BatchMetadata) -> Path:
    ensure_dirs()
    path = _path_for(meta.provider, meta.batch_id)
    path.write_text(meta.model_dump_json(indent=2))
    return path


def load(batch_id: str) -> BatchMetadata:
    """Find a batch by id regardless of which provider prefixed the file."""
    ensure_dirs()
    matches = list(BATCHES_DIR.glob(f"*_{batch_id}.json"))
    if not matches:
        raise FileNotFoundError(f"No batch found locally with id={batch_id}")
    return BatchMetadata.model_validate_json(matches[0].read_text())


def list_all() -> list[BatchMetadata]:
    ensure_dirs()
    out = []
    for p in sorted(BATCHES_DIR.glob("*.json")):
        try:
            out.append(BatchMetadata.model_validate_json(p.read_text()))
        except Exception:
            continue
    return out


def delete(batch_id: str) -> None:
    for p in BATCHES_DIR.glob(f"*_{batch_id}.json"):
        p.unlink()
