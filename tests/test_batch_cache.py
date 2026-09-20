"""Tests for batch-mode caching."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from loom.core import orchestrator
from loom.core.models import BatchMetadata, PromptItem
from loom.utils.cache import ResponseCache
from loom.utils import storage


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path / "batches")
    monkeypatch.setattr(storage, "INPUTS_DIR", tmp_path / "inputs")


def test_submit_items_skips_cached_prompts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = ResponseCache(cache_dir=tmp_path / "cache")
    cache.set("openai", "m", "cached prompt", "cached answer")

    provider = MagicMock()
    provider.submit.return_value = "batch-1"
    monkeypatch.setattr(orchestrator, "get_provider", lambda *_a, **_k: provider)
    monkeypatch.setattr(orchestrator, "resolve_api_key", lambda *_a, **_k: "fake")

    items = [
        PromptItem(custom_id="a", prompt="cached prompt"),
        PromptItem(custom_id="b", prompt="fresh prompt"),
    ]
    result = orchestrator.submit_items(
        items,
        "openai",
        "m",
        cache=cache,
        original_file_path="/tmp/x.json",
        persist=True,
    )
    assert not result.from_cache
    assert result.meta.cached_responses == {"a": "cached answer"}
    assert [p.custom_id for p in result.pending] == ["b"]
    # Only the miss was submitted.
    submitted = provider.submit.call_args[0][0]
    assert len(submitted) == 1
    assert submitted[0].custom_id == "b"


def test_submit_items_all_cached_skips_provider(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = ResponseCache(cache_dir=tmp_path / "cache")
    cache.set("openai", "m", "p1", "r1")
    cache.set("openai", "m", "p2", "r2")

    provider = MagicMock()
    monkeypatch.setattr(orchestrator, "get_provider", lambda *_a, **_k: provider)

    items = [
        PromptItem(custom_id="a", prompt="p1"),
        PromptItem(custom_id="b", prompt="p2"),
    ]
    result = orchestrator.submit_items(
        items, "openai", "m", cache=cache, original_file_path="/tmp/x.json"
    )
    assert result.from_cache
    assert result.meta.status == "completed"
    assert result.meta.batch_id.startswith("local-")
    assert result.meta.cached_responses == {"a": "r1", "b": "r2"}
    provider.submit.assert_not_called()


def test_run_batch_fully_cached_writes_output(tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    cache = ResponseCache(cache_dir=cache_dir)
    cache.set("openai", "m", "hello", "world")

    inp = tmp_path / "in.json"
    inp.write_text('[{"id": "p1", "prompt": "hello"}]')
    out = tmp_path / "out.json"

    meta = orchestrator.run_batch(
        file_path=inp,
        provider_name="openai",
        model="m",
        output_path=out,
        use_cache=True,
        cache_dir=cache_dir,
        force=True,
    )
    assert meta.batch_id.startswith("local-")
    assert meta.status == "completed"
    assert out.exists()
    assert "world" in out.read_text()


def test_fetch_batch_writes_downloaded_to_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache"
    inp = tmp_path / "in.json"
    inp.write_text('[{"id": "p1", "prompt": "hello"}, {"id": "p2", "prompt": "bye"}]')

    meta = BatchMetadata(
        batch_id="batch-99",
        provider="openai",
        model="m",
        original_file_path=str(inp),
        file_type="json",
        id_map={"p1": "p1", "p2": "p2"},
        status="validating",
        output_path=str(tmp_path / "out.json"),
        cached_responses={"p1": "from-cache"},
        cache_dir=str(cache_dir),
        source="file",
    )
    storage.save_batch(meta)

    provider = MagicMock()
    provider.check_status.return_value = "completed"
    provider.download_results.return_value = ({"p2": "from-api"}, {})
    provider.batch_error_message.return_value = None
    monkeypatch.setattr(orchestrator, "get_provider", lambda *_a, **_k: provider)
    monkeypatch.setattr(orchestrator, "resolve_api_key", lambda *_a, **_k: "fake")

    meta2, done, errors, responses = orchestrator.fetch_batch("batch-99", force=True)
    assert done
    assert responses == {"p1": "from-cache", "p2": "from-api"}
    assert errors == {}

    cache = ResponseCache(cache_dir=cache_dir)
    assert cache.get("openai", "m", "bye") == "from-api"
    # Cached-at-submit response is not re-written necessarily, but download is.
    assert (tmp_path / "out.json").exists()
