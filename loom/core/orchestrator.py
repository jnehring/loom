"""Top-level orchestration: convert input, submit, persist, fetch, merge."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..core.models import BatchMetadata, ProviderName
from ..providers import get_provider
from ..utils import converters, storage
from ..utils.keys import resolve_api_key


def run_batch(
    file_path: Path,
    provider_name: ProviderName,
    model: str,
    column: Optional[str] = None,
    api_key: Optional[str] = None,
    output_path: Optional[Path] = None,
) -> BatchMetadata:
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    ext = file_path.suffix.lower()
    if ext == ".csv":
        if not column:
            raise ValueError("--col is required for CSV input.")
        items, _df, id_map = converters.load_csv(file_path, column)
        file_type = "csv"
        prompt_column = column
    elif ext == ".json":
        items, id_map = converters.load_json(file_path)
        file_type = "json"
        prompt_column = None
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .json or .csv.")

    if not items:
        raise ValueError("Input file contains no prompts.")

    key = resolve_api_key(provider_name, api_key)
    provider = get_provider(provider_name, key)
    batch_id = provider.submit(items, model)

    meta = BatchMetadata(
        batch_id=batch_id,
        provider=provider_name,
        model=model,
        original_file_path=str(file_path),
        file_type=file_type,
        prompt_column=prompt_column,
        id_map=id_map,
        status="validating",
        output_path=str(output_path) if output_path else None,
    )
    storage.save_batch(meta)
    return meta


class OutputExistsError(Exception):
    """Raised when the merged output file would overwrite an existing file."""

    def __init__(self, meta: BatchMetadata, out_path: Path) -> None:
        super().__init__(f"Output file already exists: {out_path}")
        self.meta = meta
        self.out_path = out_path


def fetch_batch(
    batch_id: str,
    api_key: Optional[str] = None,
    keep: bool = False,
    force: bool = False,
) -> tuple[BatchMetadata, bool]:
    """Return (metadata, done). If done, results have been merged to output_path.

    On successful completion the stored metadata file in ~/.loom/batches/ is
    deleted, unless ``keep=True`` is passed.

    If the target output file already exists and ``force=False``, raises
    :class:`OutputExistsError` BEFORE downloading results — the caller (CLI)
    can prompt the user and retry with ``force=True``.
    """
    meta = storage.load_batch(batch_id)
    key = resolve_api_key(meta.provider, api_key)
    provider = get_provider(meta.provider, key)
    status = provider.check_status(meta.batch_id)
    meta.status = status

    if status != "completed":
        storage.save_batch(meta)
        return meta, False

    original = Path(meta.original_file_path)
    out_path = Path(meta.output_path) if meta.output_path else converters.default_output_path(original)

    if out_path.exists() and not force:
        raise OutputExistsError(meta, out_path)

    responses = provider.download_results(meta.batch_id)

    if meta.file_type == "json":
        converters.merge_json(original, responses, out_path)
    else:
        converters.merge_csv(original, meta.id_map, responses, out_path)

    meta.output_path = str(out_path)

    if keep:
        storage.save_batch(meta)
    else:
        storage.delete_batch(meta.batch_id)
    return meta, True


def fetch_all(
    api_key: Optional[str] = None,
    keep: bool = False,
    force: bool = False,
    on_conflict=None,
) -> list[tuple[BatchMetadata, bool]]:
    """Fetch every pending batch.

    ``on_conflict`` is an optional callable ``(meta, out_path) -> bool`` invoked
    when the merged output file already exists. Return True to overwrite,
    False to skip. If not provided and ``force=False``, conflicts are skipped.
    """
    results = []
    for meta in storage.list_batches():
        if meta.status in {"completed", "failed", "expired", "cancelled"} and meta.output_path:
            continue
        try:
            results.append(
                fetch_batch(meta.batch_id, api_key=api_key, keep=keep, force=force)
            )
        except OutputExistsError as exc:
            overwrite = bool(on_conflict(exc.meta, exc.out_path)) if on_conflict else False
            if overwrite:
                results.append(
                    fetch_batch(meta.batch_id, api_key=api_key, keep=keep, force=True)
                )
            else:
                results.append((exc.meta, False))
        except Exception:  # noqa: BLE001
            results.append((meta, False))
            meta.status = "unknown"
    return results
