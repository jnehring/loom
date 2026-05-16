"""Load JSON/CSV inputs and merge LLM responses back into the original format.

Accepts plain ``.json`` / ``.csv`` files as well as their gzip-compressed
variants ``.json.gz`` / ``.csv.gz``. Output files are always written
uncompressed.
"""

from __future__ import annotations

import gzip
import json
import re
import uuid
from pathlib import Path
from typing import IO, Tuple

import pandas as pd

from ..core.models import PromptItem


# ---------- Input format detection ----------

_INPUT_EXTS = {".json", ".csv", ".json.gz", ".csv.gz"}


def detect_format(path: Path) -> tuple[str, bool]:
    """Return ``(format, is_gzipped)`` for ``path``.

    ``format`` is one of ``"json"`` or ``"csv"``. Raises ``ValueError`` if the
    file extension is not recognised.
    """
    name = path.name.lower()
    if name.endswith(".json.gz"):
        return "json", True
    if name.endswith(".csv.gz"):
        return "csv", True
    if name.endswith(".json"):
        return "json", False
    if name.endswith(".csv"):
        return "csv", False
    raise ValueError(
        f"Unsupported file type: {path.name}. "
        f"Use one of: {sorted(_INPUT_EXTS)}."
    )


def _open_text(path: Path) -> IO[str]:
    """Open ``path`` as text, transparently handling ``.gz``."""
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


# ---------- Loaders ----------

def load_json(path: Path) -> tuple[list[PromptItem], dict[str, str]]:
    """Load a JSON list of {id, prompt}. Accepts ``.json`` or ``.json.gz``.

    Returns (prompt_items, id_map) where id_map maps custom_id -> original id.
    The custom_id sent to providers IS the original id (it must be unique).
    """
    with _open_text(Path(path)) as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError("JSON input must be a list of {id, prompt} objects.")
    items: list[PromptItem] = []
    id_map: dict[str, str] = {}
    seen: set[str] = set()
    for i, obj in enumerate(raw):
        if not isinstance(obj, dict) or "prompt" not in obj:
            raise ValueError(f"Entry {i} missing 'prompt' field.")
        orig_id = str(obj.get("id", f"row-{i}"))
        if orig_id in seen:
            raise ValueError(f"Duplicate id '{orig_id}' in JSON input.")
        seen.add(orig_id)
        items.append(PromptItem(custom_id=orig_id, prompt=str(obj["prompt"])))
        id_map[orig_id] = orig_id
    return items, id_map


def load_csv(path: Path, col: str) -> tuple[list[PromptItem], pd.DataFrame, dict[str, str]]:
    """Load a CSV; assign a UUID per row as custom_id. Accepts ``.csv`` or ``.csv.gz``.

    Returns (prompt_items, dataframe, id_map) where id_map maps custom_id -> str(row_index).
    """
    # pandas auto-detects gzip from the .gz suffix.
    df = pd.read_csv(path)
    if col not in df.columns:
        raise ValueError(
            f"Column '{col}' not found in {path}. Available: {list(df.columns)}"
        )
    items: list[PromptItem] = []
    id_map: dict[str, str] = {}
    for idx, value in df[col].items():
        cid = f"row-{idx}-{uuid.uuid4().hex[:8]}"
        items.append(PromptItem(custom_id=cid, prompt=str(value)))
        id_map[cid] = str(idx)
    return items, df, id_map


# ---------- Mergers ----------

def merge_json(
    original_path: Path,
    responses: dict[str, str],
    output_path: Path,
    *,
    with_meta: bool = False,
    provider: str | None = None,
    model: str | None = None,
) -> Path:
    """Re-load the original JSON and attach llm_response to each entry by id.

    If ``with_meta`` is True, also add ``llm_provider`` and ``llm_model`` fields.
    """
    with _open_text(Path(original_path)) as f:
        raw = json.load(f)
    for i, obj in enumerate(raw):
        orig_id = str(obj.get("id", f"row-{i}"))
        obj["llm_response"] = responses.get(orig_id, None)
        if with_meta:
            obj["llm_provider"] = provider
            obj["llm_model"] = model
    output_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False))
    return output_path


def merge_csv(
    original_path: Path,
    id_map: dict[str, str],
    responses: dict[str, str],
    output_path: Path,
    *,
    with_meta: bool = False,
    provider: str | None = None,
    model: str | None = None,
) -> Path:
    """Add llm_response column, preserving original row order.

    If ``with_meta`` is True, also add ``llm_provider`` and ``llm_model`` columns.
    """
    df = pd.read_csv(original_path)
    # Build per-row response by inverting id_map (row_index -> custom_id).
    row_to_resp: dict[int, str] = {}
    for cid, row_idx_str in id_map.items():
        row_to_resp[int(row_idx_str)] = responses.get(cid, "")
    df["llm_response"] = df.index.map(lambda i: row_to_resp.get(i, ""))
    if with_meta:
        df["llm_provider"] = provider
        df["llm_model"] = model
    df.to_csv(output_path, index=False)
    return output_path


# ---------- Output path naming ----------

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_for_filename(s: str) -> str:
    """Replace any character that is not alphanumeric / dot / underscore / dash."""
    return _UNSAFE_FILENAME.sub("_", s).strip("_") or "unknown"


def default_output_path(
    input_path: Path,
    provider: str | None = None,
    model: str | None = None,
) -> Path:
    """Compute the default output file path.

    The output is always uncompressed (``.json`` or ``.csv``); any ``.gz``
    suffix on the input is stripped. ``provider`` and ``model`` are appended
    to the stem when supplied, e.g.::

        data.csv, provider=openai, model=gpt-4o-mini
            -> data_results_openai_gpt-4o-mini.csv

        data.json.gz, provider=openrouter, model=openai/gpt-4o-mini
            -> data_results_openrouter_openai_gpt-4o-mini.json
    """
    p = Path(input_path)
    name_lower = p.name.lower()
    if name_lower.endswith(".json.gz"):
        stem = p.name[: -len(".json.gz")]
        ext = ".json"
    elif name_lower.endswith(".csv.gz"):
        stem = p.name[: -len(".csv.gz")]
        ext = ".csv"
    else:
        stem = p.stem
        ext = p.suffix
    parts = [stem, "results"]
    if provider:
        parts.append(sanitize_for_filename(provider))
    if model:
        parts.append(sanitize_for_filename(model))
    return p.with_name("_".join(parts) + ext)
