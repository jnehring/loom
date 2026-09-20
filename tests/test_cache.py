"""Tests for ResponseCache and path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from loom.utils.cache import ResponseCache, get_cache, reset_default_cache
from loom.utils import paths


def test_response_cache_roundtrip(tmp_path: Path) -> None:
    cache = ResponseCache(cache_dir=tmp_path / "c")
    assert cache.get("openai", "m", "hello") is None
    cache.set("openai", "m", "hello", "world")
    assert cache.get("openai", "m", "hello") == "world"
    assert cache.count() == 1


def test_response_cache_key_depends_on_all_parts(tmp_path: Path) -> None:
    cache = ResponseCache(cache_dir=tmp_path)
    cache.set("openai", "m1", "p", "a")
    assert cache.get("openai", "m2", "p") is None
    assert cache.get("anthropic", "m1", "p") is None
    assert cache.get("openai", "m1", "other") is None


def test_disabled_cache_is_noop(tmp_path: Path) -> None:
    cache = ResponseCache(cache_dir=tmp_path, enabled=False)
    cache.set("openai", "m", "p", "r")
    assert cache.get("openai", "m", "p") is None
    assert cache.count() == 0
    assert cache.clear() == 0


def test_clear_removes_entries(tmp_path: Path) -> None:
    cache = ResponseCache(cache_dir=tmp_path)
    cache.set("openai", "m", "a", "1")
    cache.set("openai", "m", "b", "2")
    assert cache.clear() == 2
    assert cache.count() == 0


def test_default_cache_dir_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOOM_CACHE_DIR", str(tmp_path / "custom"))
    assert paths.default_cache_dir() == tmp_path / "custom"


def test_loom_home_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LOOM_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("LOOM_CACHE_DIR", raising=False)
    assert paths.loom_home() == tmp_path / "home"
    assert paths.default_cache_dir() == tmp_path / "home" / "cache"
    assert paths.default_batches_dir() == tmp_path / "home" / "batches"


def test_get_cache_with_explicit_dir(tmp_path: Path) -> None:
    reset_default_cache()
    c = get_cache(cache_dir=tmp_path / "x")
    assert c.dir == tmp_path / "x"
    c.set("openai", "m", "p", "r")
    assert (tmp_path / "x").exists()
