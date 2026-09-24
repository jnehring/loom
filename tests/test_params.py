"""Generation settings (temperature, max_tokens, …): translation per provider, cache keys, API and CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from loom import GenerationParams, Loom, UnsupportedParameterError
from loom.core import orchestrator
from loom.core.models import BatchMetadata, PromptItem
from loom.main import app
from loom.providers.anthropic import AnthropicBatchProvider
from loom.providers.anthropic_sync import AnthropicSyncProvider
from loom.providers.google import GoogleBatchProvider
from loom.providers.google_sync import GoogleSyncProvider
from loom.providers.openai import OpenAIBatchProvider
from loom.providers.openai_sync import OpenAISyncProvider
from loom.providers.openrouter_sync import OpenRouterSyncProvider
from loom.utils import storage
from loom.utils.cache import ResponseCache

FULL = GenerationParams(temperature=0.2, max_tokens=300, top_p=0.9, stop=["END"], system="Be brief.")
ITEMS = [PromptItem(custom_id="c1", prompt="hi")]


@pytest.fixture(autouse=True)
def _isolate_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path / "batches")
    monkeypatch.setattr(storage, "INPUTS_DIR", tmp_path / "inputs")


# ---------- GenerationParams ----------

def test_defaults_are_empty_and_validated() -> None:
    assert GenerationParams().is_default()
    assert GenerationParams().cache_token() == ""
    with pytest.raises(ValueError):
        GenerationParams(temperature=5)
    with pytest.raises(ValueError):
        GenerationParams(max_tokens=0)
    with pytest.raises(ValueError):
        GenerationParams(tempreature=0.2)  # typo: unknown fields are rejected


def test_cache_token_is_canonical() -> None:
    a = GenerationParams(temperature=0.2, max_tokens=10)
    b = GenerationParams(max_tokens=10, temperature=0.2)
    assert a.cache_token() == b.cache_token() == '{"max_tokens":10,"temperature":0.2}'


def test_merged_overrides_only_set_fields() -> None:
    base = GenerationParams(temperature=0.2, max_tokens=100)
    out = base.merged(GenerationParams(max_tokens=50))
    assert (out.temperature, out.max_tokens) == (0.2, 50)


def test_old_batch_metadata_without_params_loads() -> None:
    meta = BatchMetadata.model_validate_json(json.dumps({
        "batch_id": "b", "provider": "google", "model": "m", "original_file_path": "", "file_type": "json"}))
    assert meta.params.is_default()


# ---------- Cache ----------

def test_cache_key_unchanged_without_settings(tmp_path: Path) -> None:
    """Keys from before generation settings existed stay valid."""
    assert ResponseCache.key("p", "m", "x") == ResponseCache.key("p", "m", "x", GenerationParams())


def test_cache_separates_settings(tmp_path: Path) -> None:
    cache = ResponseCache(cache_dir=tmp_path)
    cache.set("google", "m", "x", "default")
    cache.set("google", "m", "x", "cold", GenerationParams(temperature=0))
    assert cache.get("google", "m", "x") == "default"
    assert cache.get("google", "m", "x", GenerationParams(temperature=0)) == "cold"
    assert cache.get("google", "m", "x", GenerationParams(temperature=1)) is None
    stored = [json.loads(p.read_text()) for p in tmp_path.glob("*.json")]
    assert {"temperature": 0.0} in [s.get("params") for s in stored]


# ---------- Providers: request fields ----------

def test_openai_batch_body() -> None:
    provider = OpenAIBatchProvider.__new__(OpenAIBatchProvider)
    params = FULL.merged(GenerationParams(seed=7, json_mode=True, extra={"reasoning_effort": "low"}))
    line = json.loads(provider._build_jsonl(ITEMS, "gpt", params))
    body = line["body"]
    assert body["messages"] == [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hi"}]
    assert body["temperature"] == 0.2 and body["max_completion_tokens"] == 300 and "max_tokens" not in body
    assert body["top_p"] == 0.9 and body["stop"] == ["END"] and body["seed"] == 7
    assert body["response_format"] == {"type": "json_object"} and body["reasoning_effort"] == "low"


def test_openai_without_settings_sends_only_messages() -> None:
    provider = OpenAIBatchProvider.__new__(OpenAIBatchProvider)
    body = json.loads(provider._build_jsonl(ITEMS, "gpt"))["body"]
    assert set(body) == {"model", "messages"}


def test_openai_rejects_top_k() -> None:
    provider = OpenAIBatchProvider.__new__(OpenAIBatchProvider)
    with pytest.raises(UnsupportedParameterError, match="top_k"):
        provider._build_jsonl(ITEMS, "gpt", GenerationParams(top_k=5))


def test_openai_and_openrouter_sync() -> None:
    for cls, max_key in ((OpenAISyncProvider, "max_completion_tokens"), (OpenRouterSyncProvider, "max_tokens")):
        provider = cls.__new__(cls)
        provider.client = MagicMock()
        provider.client.chat.completions.create.return_value.choices = [MagicMock()]
        provider.client.chat.completions.create.return_value.choices[0].message.content = "ok"
        params = FULL if cls is OpenAISyncProvider else FULL.merged(GenerationParams(top_k=4))
        assert provider.generate("hi", "m", params) == "ok"
        kwargs = provider.client.chat.completions.create.call_args.kwargs
        assert kwargs["temperature"] == 0.2 and kwargs[max_key] == 300
        assert kwargs["messages"][0] == {"role": "system", "content": "Be brief."}
        if cls is OpenRouterSyncProvider:
            assert kwargs["extra_body"] == {"top_k": 4}


def test_anthropic_batch_and_sync() -> None:
    batch = AnthropicBatchProvider.__new__(AnthropicBatchProvider)
    batch.client = MagicMock()
    batch.client.messages.batches.create.return_value.id = "msgbatch_1"
    assert batch.submit(ITEMS, "claude", FULL.merged(GenerationParams(top_k=3))) == "msgbatch_1"
    req = batch.client.messages.batches.create.call_args.kwargs["requests"][0]
    params = req["params"]
    assert params["max_tokens"] == 300 and params["temperature"] == 0.2 and params["top_k"] == 3
    assert params["stop_sequences"] == ["END"] and params["system"] == "Be brief."

    sync = AnthropicSyncProvider.__new__(AnthropicSyncProvider)
    sync.client = MagicMock()
    sync.client.messages.create.return_value.content = [MagicMock(text="ok")]
    assert sync.generate("hi", "claude") == "ok"
    assert sync.client.messages.create.call_args.kwargs["max_tokens"] == 4096  # Anthropic requires it


@pytest.mark.parametrize("name", ["seed", "presence_penalty", "frequency_penalty", "json_mode"])
def test_anthropic_rejects_unsupported(name: str) -> None:
    sync = AnthropicSyncProvider.__new__(AnthropicSyncProvider)
    sync.client = MagicMock()
    value = True if name == "json_mode" else 1
    with pytest.raises(UnsupportedParameterError, match=name):
        sync.generate("hi", "claude", GenerationParams(**{name: value}))


def test_google_batch_inline_config() -> None:
    provider = GoogleBatchProvider.__new__(GoogleBatchProvider)
    provider.client = MagicMock()
    provider.client.batches.create.return_value.name = "batches/1"
    params = FULL.merged(GenerationParams(top_k=5, seed=1, json_mode=True, extra={"thinking_config": {"thinking_budget": 0}}))
    provider.submit(ITEMS, "gemini", params)
    request = provider.client.batches.create.call_args.kwargs["src"][0]
    config = request["config"]
    assert config["temperature"] == 0.2 and config["max_output_tokens"] == 300 and config["top_k"] == 5
    assert config["stop_sequences"] == ["END"] and config["seed"] == 1
    assert config["system_instruction"] == {"parts": [{"text": "Be brief."}]}
    assert config["response_mime_type"] == "application/json"
    assert config["thinking_config"] == {"thinking_budget": 0}

    provider.submit(ITEMS, "gemini")
    assert "config" not in provider.client.batches.create.call_args.kwargs["src"][0]


def test_google_config_is_accepted_by_sdk() -> None:
    """The inline dict must validate as the SDK's InlinedRequest (catches renamed fields)."""
    from google.genai import types

    provider = GoogleBatchProvider.__new__(GoogleBatchProvider)
    provider.client = MagicMock()
    provider.submit(ITEMS, "gemini", FULL.merged(GenerationParams(top_k=5, seed=1, json_mode=True)))
    request = provider.client.batches.create.call_args.kwargs["src"][0]
    parsed = types.InlinedRequest.model_validate(request)
    assert parsed.config.temperature == 0.2 and parsed.config.max_output_tokens == 300


def test_google_sync_passes_config() -> None:
    provider = GoogleSyncProvider.__new__(GoogleSyncProvider)
    provider.client = MagicMock()
    provider.client.models.generate_content.return_value.text = "ok"
    assert provider.generate("hi", "gemini", GenerationParams(temperature=0.1)) == "ok"
    assert provider.client.models.generate_content.call_args.kwargs["config"] == {"temperature": 0.1}
    provider.generate("hi", "gemini")
    assert "config" not in provider.client.models.generate_content.call_args.kwargs


# ---------- Orchestrator and API ----------

def test_submit_and_fetch_use_settings_for_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = ResponseCache(cache_dir=tmp_path / "cache")
    cache.set("google", "m", "hi", "answer at default settings")
    provider = MagicMock()
    provider.submit.return_value = "batches/9"
    monkeypatch.setattr(orchestrator, "get_provider", lambda *_a, **_k: provider)
    monkeypatch.setattr(orchestrator, "resolve_api_key", lambda *_a, **_k: "fake")

    params = GenerationParams(temperature=0)
    client = Loom("google", "m", api_key="fake", cache_dir=tmp_path / "cache", temperature=0)
    job = client.submit(["hi"])
    # the default-settings answer must not be reused for temperature 0
    assert not job.from_cache
    assert provider.submit.call_args.args[2] == params
    assert storage.load_batch("batches/9").params == params

    provider.check_status.return_value = "completed"
    provider.download_results.return_value = ({"prompt-0": "cold answer"}, {})
    provider.batch_error_message.return_value = None
    result = job.fetch(keep=True)
    assert result.done and result.responses == {"prompt-0": "cold answer"}
    assert cache.get("google", "m", "hi", params) == "cold answer"
    assert cache.get("google", "m", "hi") == "answer at default settings"


def test_unsupported_setting_fails_before_submit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = MagicMock()
    monkeypatch.setattr(orchestrator, "get_provider", lambda *_a, **_k: provider)
    monkeypatch.setattr(orchestrator, "resolve_api_key", lambda *_a, **_k: "fake")
    with pytest.raises(UnsupportedParameterError):
        Loom("openai", "m", api_key="fake", cache_dir=tmp_path, top_k=3).submit(["hi"])
    provider.submit.assert_not_called()


def test_client_defaults_and_per_call_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    provider = MagicMock()
    provider.generate.side_effect = lambda prompt, model, params=None: json.dumps(params.set_fields() if params else {})
    monkeypatch.setattr(orchestrator, "get_sync_provider", lambda *_a, **_k: provider)
    client = Loom("google", "m", api_key="fake", cache_dir=tmp_path, temperature=0.2, max_tokens=100)
    assert json.loads(client.generate("a")) == {"temperature": 0.2, "max_tokens": 100}
    assert json.loads(client.generate("a", params={"max_tokens": 20})) == {"temperature": 0.2, "max_tokens": 20}
    # without settings the provider is called with two arguments (older custom providers keep working)
    Loom("google", "m", api_key="fake", cache_dir=tmp_path).generate("b")
    assert provider.generate.call_args.args == ("b", "m")


def test_cli_run_sync_passes_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    inp = tmp_path / "in.json"
    inp.write_text('[{"id": "1", "prompt": "hi"}]')
    seen = {}

    def fake_generate_sync(**kwargs):
        seen.update(kwargs)
        return tmp_path / "out.json", 1, 0, 0, {}

    monkeypatch.setattr(orchestrator, "generate_sync", fake_generate_sync)
    result = CliRunner().invoke(app, [
        "run", "-f", str(inp), "-p", "google", "-m", "gemini", "--sync", "--temperature", "0.3", "--max-tokens", "50",
        "--stop", "A", "--stop", "B", "--system", "S", "--json", "--param", 'thinking_config={"thinking_budget": 0}',
    ])
    assert result.exit_code == 0, result.stdout
    assert seen["params"] == GenerationParams(temperature=0.3, max_tokens=50, stop=["A", "B"], system="S",
                                              json_mode=True, extra={"thinking_config": {"thinking_budget": 0}})


def test_cli_rejects_invalid_setting(tmp_path: Path) -> None:
    inp = tmp_path / "in.json"
    inp.write_text('[{"id": "1", "prompt": "hi"}]')
    result = CliRunner().invoke(app, ["run", "-f", str(inp), "-p", "google", "-m", "g", "--sync", "--temperature", "9"])
    assert result.exit_code == 1
    assert "invalid generation setting" in result.stdout
