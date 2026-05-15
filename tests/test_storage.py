"""Tests for ~/.loom/batches/ persistence (redirected to a tmp dir)."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.core.models import BatchMetadata
from loom.utils import storage


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path / "batches")


def _meta(batch_id: str = "b1", provider: str = "openai") -> BatchMetadata:
    return BatchMetadata(
        batch_id=batch_id,
        provider=provider,
        model="m",
        original_file_path="/tmp/x.json",
        file_type="json",
    )


def test_save_load_roundtrip() -> None:
    storage.save_batch(_meta("b1"))
    loaded = storage.load_batch("b1")
    assert loaded.batch_id == "b1"
    assert loaded.provider == "openai"


def test_list_returns_newest_first() -> None:
    storage.save_batch(_meta("a"))
    storage.save_batch(_meta("b", provider="anthropic"))
    items = storage.list_batches()
    assert {m.batch_id for m in items} == {"a", "b"}


def test_delete_batch_removes_file() -> None:
    storage.save_batch(_meta("zap"))
    assert storage.delete_batch("zap") is True
    with pytest.raises(FileNotFoundError):
        storage.load_batch("zap")


def test_delete_batch_returns_false_when_missing() -> None:
    assert storage.delete_batch("nope") is False
