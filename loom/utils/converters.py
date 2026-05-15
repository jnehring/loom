"""Load JSON/CSV inputs and merge LLM responses back into the original format."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Tuple

import pandas as pd

from ..core.models import PromptItem


# ---------- Loaders ----------

def load_json(path: Path) -> tuple[list[PromptItem], dict[str, str]]:
    """Load a JSON list of {id, prompt}.

    Returns (prompt_items, id_map) where id_map maps custom_id -> original id.
    The custom_id sent to providers IS the original id (it must be unique).
    """
    raw = json.loads(Path(path).read_text())
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
    """Load a CSV; assign a UUID per row as custom_id.

    Returns (prompt_items, dataframe, id_map) where id_map maps custom_id -> str(row_index).
    """
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
) -> Path:
    """Re-load the original JSON and attach llm_response to each entry by id."""
    raw = json.loads(Path(original_path).read_text())
    for i, obj in enumerate(raw):
        orig_id = str(obj.get("id", f"row-{i}"))
        obj["llm_response"] = responses.get(orig_id, None)
    output_path.write_text(json.dumps(raw, indent=2, ensure_ascii=False))
    return output_path


def merge_csv(
    original_path: Path,
    id_map: dict[str, str],
    responses: dict[str, str],
    output_path: Path,
) -> Path:
    """Add llm_response column, preserving original row order."""
    df = pd.read_csv(original_path)
    # Build per-row response by inverting id_map (row_index -> custom_id).
    row_to_resp: dict[int, str] = {}
    for cid, row_idx_str in id_map.items():
        row_to_resp[int(row_idx_str)] = responses.get(cid, "")
    df["llm_response"] = df.index.map(lambda i: row_to_resp.get(i, ""))
    df.to_csv(output_path, index=False)
    return output_path


def default_output_path(input_path: Path) -> Path:
    p = Path(input_path)
    stem = p.stem
    return p.with_name(f"{stem}_results{p.suffix}")
