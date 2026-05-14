from __future__ import annotations

import uuid
from pathlib import Path

from ..providers import get_provider
from ..providers.base import BatchProvider
from ..utils import storage
from ..utils.converters import (
    load_prompts_csv,
    load_prompts_json,
    write_csv_results,
    write_json_results,
)
from .models import BatchMetadata, PromptItem


def submit_batch(
    file_path: Path,
    provider_name: str,
    model: str,
    column: str | None = None,
    api_key: str | None = None,
) -> BatchMetadata:
    """Load prompts, submit to provider, persist metadata. Returns the metadata."""
    file_type = _detect_file_type(file_path)
    prompts, id_map = _load_with_custom_ids(file_path, file_type, column)

    provider = get_provider(provider_name, api_key=api_key)
    batch_id = provider.submit(prompts, model)

    meta = BatchMetadata(
        batch_id=batch_id,
        provider=provider.name,  # type: ignore[arg-type]
        model=model,
        original_file_path=str(file_path.resolve()),
        file_type=file_type,
        prompt_column=column,
        id_map=id_map,
    )
    storage.save(meta)
    return meta


def fetch_batch(batch_id: str, api_key: str | None = None) -> tuple[BatchMetadata, Path | None]:
    """Poll provider; if complete, write merged results next to the original input."""
    meta = storage.load(batch_id)
    provider = get_provider(meta.provider, api_key=api_key)
    status = provider.status(batch_id)

    if status.state != "completed":
        return meta, None

    raw = provider.fetch_results(batch_id).responses
    # Translate provider custom_ids back to the original-file identity.
    mapped: dict[str, str] = {}
    for custom_id, text in raw.items():
        original_id = meta.id_map.get(custom_id, custom_id)
        mapped[original_id] = text if text is not None else ""

    input_path = Path(meta.original_file_path)
    if meta.file_type == "json":
        out = write_json_results(input_path, mapped, meta.provider, meta.model)
    else:
        assert meta.prompt_column, "CSV batch missing prompt_column"
        out = write_csv_results(input_path, meta.prompt_column, mapped, meta.provider, meta.model)
    return meta, out


def _detect_file_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return "json"
    if suffix == ".csv":
        return "csv"
    raise ValueError(f"Unsupported file type: {suffix}. Use .json or .csv.")


def _load_with_custom_ids(
    file_path: Path,
    file_type: str,
    column: str | None,
) -> tuple[list[PromptItem], dict[str, str]]:
    """Load prompts and remap each to a unique custom_id, returning the id_map.

    For JSON: id_map[custom_id] = original_id (so duplicates in the input are tolerated).
    For CSV:  id_map[custom_id] = row_index_as_string.
    """
    id_map: dict[str, str] = {}
    if file_type == "json":
        items = load_prompts_json(file_path)
    else:
        if not column:
            raise ValueError("CSV input requires --col to specify the prompt column.")
        items, _, _ = load_prompts_csv(file_path, column)

    remapped: list[PromptItem] = []
    for item in items:
        cid = f"loom-{uuid.uuid4().hex[:12]}"
        id_map[cid] = item.id
        remapped.append(PromptItem(id=cid, prompt=item.prompt))
    return remapped, id_map
