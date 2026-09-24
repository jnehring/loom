"""Tests for the public Loom library API."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from loom import Loom, generate, generate_many
from loom.api import GenerationResult


@pytest.fixture
def fake_sync(monkeypatch: pytest.MonkeyPatch):
    provider = MagicMock()
    provider.generate.side_effect = lambda prompt, model: f"echo:{prompt}"
    provider.count_tokens.side_effect = lambda prompt, model: len(prompt.split())
    monkeypatch.setattr(
        "loom.core.orchestrator.get_sync_provider",
        lambda *_a, **_k: provider,
    )
    return provider


def test_generate_many_preserves_order(fake_sync, tmp_path: Path) -> None:
    client = Loom("openai", "m", api_key="fake", cache_dir=tmp_path / "cache")
    result = client.generate_many(["one", "two", "three"])
    assert isinstance(result, GenerationResult)
    assert result.texts == ["echo:one", "echo:two", "echo:three"]
    assert result.total == 3
    assert result.cache_hits == 0
    assert result.errors == 0
    assert fake_sync.generate.call_count == 3


def test_generate_many_uses_cache(fake_sync, tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    client = Loom("openai", "m", api_key="fake", cache_dir=cache_dir)
    client.generate_many(["a", "b"])
    assert fake_sync.generate.call_count == 2

    result = client.generate_many(["a", "b", "c"])
    assert result.cache_hits == 2
    assert result.texts == ["echo:a", "echo:b", "echo:c"]
    assert fake_sync.generate.call_count == 3  # only "c" is new


def test_generate_many_captures_errors(fake_sync, tmp_path: Path) -> None:
    # Keyed by prompt, not call order: prompts run concurrently
    def answer(prompt: str, model: str) -> str:
        if prompt == "b":
            raise RuntimeError("boom")
        return {"a": "ok", "c": "ok2"}[prompt]

    fake_sync.generate.side_effect = answer
    client = Loom("openai", "m", api_key="fake", cache_dir=tmp_path, use_cache=False)
    result = client.generate_many(["a", "b", "c"])
    assert result.errors == 1
    assert result.texts[0] == "ok"
    assert result.texts[1] == ""
    assert result.results[1].error == "boom"
    assert result.texts[2] == "ok2"


def test_generate_single(fake_sync, tmp_path: Path) -> None:
    client = Loom("openai", "m", api_key="fake", cache_dir=tmp_path)
    assert client.generate("hi") == "echo:hi"


def test_generate_frame(fake_sync, tmp_path: Path) -> None:
    client = Loom("openai", "m", api_key="fake", cache_dir=tmp_path, with_meta=True)
    df = pd.DataFrame({"text": ["x", "y"], "id": [1, 2]})
    out = client.generate_frame(df)
    assert list(out["llm_response"]) == ["echo:x", "echo:y"]
    assert list(out["llm_provider"]) == ["openai", "openai"]
    assert list(out["llm_model"]) == ["m", "m"]
    assert list(out["id"]) == [1, 2]


def test_run_file(fake_sync, tmp_path: Path) -> None:
    inp = tmp_path / "in.json"
    inp.write_text('[{"id": "p1", "prompt": "hello"}]')
    client = Loom("openai", "m", api_key="fake", cache_dir=tmp_path / "c")
    run = client.run_file(inp, force=True)
    assert run.total == 1
    assert run.output_path.exists()
    assert "hello" in run.output_path.read_text() or "echo:hello" in run.output_path.read_text()


def test_count_tokens(fake_sync, tmp_path: Path) -> None:
    client = Loom("anthropic", "m", api_key="fake", cache_dir=tmp_path)
    result = client.count_tokens(["one two", "three"])
    assert result.total_tokens == 3  # 2 + 1
    assert result.total == 2
    assert result.errors == 0


def test_module_level_generate(fake_sync, tmp_path: Path) -> None:
    assert generate("z", "openai", "m", api_key="fake", cache_dir=tmp_path) == "echo:z"
    r = generate_many(["a"], "openai", "m", api_key="fake", cache_dir=tmp_path)
    assert r.texts == ["echo:a"]


def test_lazy_exports() -> None:
    import loom

    assert loom.Loom is Loom
    assert loom.__version__
    assert "GenerationResult" in dir(loom)
