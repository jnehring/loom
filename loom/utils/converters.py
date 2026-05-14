from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Iterable

from ..core.models import PromptItem


def _safe(s: str) -> str:
    """Make a provider/model name safe for use in a filename."""
    return re.sub(r"[^A-Za-z0-9._-]+", "-", s).strip("-") or "x"


def _output_path(input_path: Path, provider: str, model: str, suffix: str) -> Path:
    name = f"{input_path.stem}_results_{_safe(provider)}_{_safe(model)}{suffix}"
    return input_path.with_name(name)


def load_prompts_json(path: Path) -> list[PromptItem]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {path}")
    return [PromptItem(**row) for row in data]


def load_prompts_csv(path: Path, column: str) -> tuple[list[PromptItem], list[dict], list[str]]:
    """Returns (prompts, original_rows, fieldnames). Uses the row index as id."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if column not in fieldnames:
            raise ValueError(f"Column '{column}' not found in CSV. Available: {fieldnames}")
        rows = list(reader)
    prompts = [PromptItem(id=str(i), prompt=row[column]) for i, row in enumerate(rows)]
    return prompts, rows, fieldnames


def write_json_results(
    input_path: Path,
    results: dict[str, str],
    provider: str,
    model: str,
) -> Path:
    """results: original_id -> llm_response."""
    data = json.loads(input_path.read_text())
    for row in data:
        row["llm_response"] = results.get(row["id"], None)
        row["llm_provider"] = provider
        row["llm_model"] = model
    out = _output_path(input_path, provider, model, ".json")
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return out


def write_csv_results(
    input_path: Path,
    column: str,
    results: dict[str, str],
    provider: str,
    model: str,
) -> Path:
    """results keyed by original row index (as string)."""
    with input_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    for col in ("llm_response", "llm_provider", "llm_model"):
        if col not in fieldnames:
            fieldnames.append(col)
    for i, row in enumerate(rows):
        row["llm_response"] = results.get(str(i), "")
        row["llm_provider"] = provider
        row["llm_model"] = model
    out = _output_path(input_path, provider, model, ".csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out


def dump_jsonl(items: Iterable[dict]) -> bytes:
    """Render an iterable of dicts as JSONL bytes."""
    return ("\n".join(json.dumps(it, ensure_ascii=False) for it in items) + "\n").encode("utf-8")


def parse_jsonl(text: str) -> list[dict]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]
