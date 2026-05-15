"""Tests for JSON/CSV load + merge — offline, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from loom.utils import converters


def test_load_json_roundtrip(tmp_path: Path) -> None:
    src = tmp_path / "in.json"
    src.write_text(json.dumps([
        {"id": "a", "prompt": "hello"},
        {"id": "b", "prompt": "world"},
    ]))

    items, id_map = converters.load_json(src)
    assert [it.custom_id for it in items] == ["a", "b"]
    assert [it.prompt for it in items] == ["hello", "world"]
    assert id_map == {"a": "a", "b": "b"}


def test_load_json_rejects_duplicate_ids(tmp_path: Path) -> None:
    src = tmp_path / "dup.json"
    src.write_text(json.dumps([
        {"id": "x", "prompt": "1"},
        {"id": "x", "prompt": "2"},
    ]))
    with pytest.raises(ValueError, match="Duplicate"):
        converters.load_json(src)


def test_merge_json_preserves_order_and_attaches_responses(tmp_path: Path) -> None:
    src = tmp_path / "in.json"
    src.write_text(json.dumps([
        {"id": "a", "prompt": "hello", "meta": "keep"},
        {"id": "b", "prompt": "world"},
    ]))
    out = tmp_path / "out.json"

    # Responses returned in reversed order — merge must still put them right.
    responses = {"b": "B_RESP", "a": "A_RESP"}
    converters.merge_json(src, responses, out)

    merged = json.loads(out.read_text())
    assert merged[0]["id"] == "a"
    assert merged[0]["llm_response"] == "A_RESP"
    assert merged[0]["meta"] == "keep"          # original fields preserved
    assert merged[1]["llm_response"] == "B_RESP"


def test_load_csv_assigns_unique_ids(tmp_path: Path) -> None:
    src = tmp_path / "in.csv"
    src.write_text("id,text,extra\n1,hello,foo\n2,world,bar\n")

    items, df, id_map = converters.load_csv(src, "text")
    assert len(items) == 2
    assert all(it.custom_id in id_map for it in items)
    # id_map values are row indices as strings
    assert sorted(id_map.values()) == ["0", "1"]
    assert list(df.columns) == ["id", "text", "extra"]


def test_load_csv_missing_column_raises(tmp_path: Path) -> None:
    src = tmp_path / "in.csv"
    src.write_text("id,text\n1,hello\n")
    with pytest.raises(ValueError, match="not found"):
        converters.load_csv(src, "nope")


def test_merge_csv_appends_response_column_and_preserves_rows(tmp_path: Path) -> None:
    src = tmp_path / "in.csv"
    src.write_text("id,text,extra\n10,hello,foo\n20,world,bar\n")

    items, _df, id_map = converters.load_csv(src, "text")
    cid_for_row0 = next(cid for cid, idx in id_map.items() if idx == "0")
    cid_for_row1 = next(cid for cid, idx in id_map.items() if idx == "1")
    responses = {cid_for_row1: "WORLD!", cid_for_row0: "HELLO!"}

    out = tmp_path / "out.csv"
    converters.merge_csv(src, id_map, responses, out)

    result = pd.read_csv(out)
    assert list(result.columns) == ["id", "text", "extra", "llm_response"]
    assert result.iloc[0]["llm_response"] == "HELLO!"
    assert result.iloc[1]["llm_response"] == "WORLD!"
    assert result.iloc[0]["extra"] == "foo"


def test_default_output_path() -> None:
    p = converters.default_output_path(Path("/tmp/data.csv"))
    assert p.name == "data_results.csv"
    p = converters.default_output_path(Path("/tmp/prompts.json"))
    assert p.name == "prompts_results.json"
