from unittest.mock import MagicMock

from loom.core import orchestrator
from loom.providers.anthropic import AnthropicBatchProvider
from loom.providers.openai import OpenAIBatchProvider
from loom.utils.errors import format_api_error, format_exception


def test_format_exception_uses_message():
    assert format_exception(ValueError("bad prompt")) == "bad prompt"


def test_format_exception_falls_back_to_type_name():
    class EmptyError(Exception):
        pass

    assert format_exception(EmptyError()) == "EmptyError"


def test_format_api_error_from_dict():
    assert format_api_error({"message": "rate limited"}) == "rate limited"


def test_openai_batch_download_results_extracts_line_errors():
    provider = OpenAIBatchProvider(api_key="fake")
    provider.client = MagicMock()

    batch = MagicMock()
    batch.output_file_id = "file-1"
    provider.client.batches.retrieve.return_value = batch

    jsonl = (
        '{"custom_id":"ok","response":{"body":{"choices":[{"message":{"content":"yes"}}]}}}\n'
        '{"custom_id":"bad","error":{"message":"model not found"}}\n'
    )
    content = MagicMock()
    content.read.return_value = jsonl.encode("utf-8")
    provider.client.files.content.return_value = content

    responses, errors = provider.download_results("batch-1")
    assert responses == {"ok": "yes", "bad": ""}
    assert errors == {"bad": "model not found"}


def test_anthropic_batch_download_results_extracts_errored_results():
    provider = AnthropicBatchProvider(api_key="fake")
    provider.client = MagicMock()

    ok = MagicMock()
    ok.custom_id = "ok"
    ok.result.type = "succeeded"
    ok.result.message.content = [MagicMock(type="text", text="hello")]

    bad = MagicMock()
    bad.custom_id = "bad"
    bad.result.type = "errored"
    bad.result.error.message = "overloaded"

    provider.client.messages.batches.results.return_value = [ok, bad]

    responses, errors = provider.download_results("batch-1")
    assert responses == {"ok": "hello", "bad": ""}
    assert errors == {"bad": "overloaded"}


def test_generate_sync_collects_error_messages(monkeypatch, tmp_path):
    input_path = tmp_path / "prompts.json"
    input_path.write_text('[{"id": "p1", "prompt": "hi"}]')

    provider = MagicMock()
    provider.generate.side_effect = RuntimeError("quota exceeded")
    monkeypatch.setattr(orchestrator, "get_sync_provider", lambda *_args, **_kwargs: provider)
    monkeypatch.setattr(orchestrator.response_cache, "get", lambda *_args, **_kwargs: None)

    out_path, total, hits, errors, error_messages = orchestrator.generate_sync(
        file_path=input_path,
        provider_name="google",
        model="gemini-test",
        api_key="fake",
        use_cache=False,
        force=True,
    )

    assert total == 1
    assert hits == 0
    assert errors == 1
    assert error_messages == {"p1": "quota exceeded"}
    assert out_path.exists()
