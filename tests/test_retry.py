"""Retries of transient provider errors and reporting of failed batches."""

from __future__ import annotations

import ssl
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import openai
import pytest
from google.genai import errors as genai_errors

from loom import BatchFailedError
from loom.api import BatchFetchResult, BatchJob
from loom.core import orchestrator
from loom.core.models import BatchMetadata, PromptItem
from loom.providers.openai import OpenAIBatchProvider
from loom.utils import retry, storage
from loom.utils.retry import is_rate_limited, is_transient, with_retries

_REQ = httpx.Request("GET", "https://api.example.com")


def _status_error(code: int) -> openai.APIStatusError:
    return openai.APIStatusError("boom", response=httpx.Response(code, request=_REQ), body=None)


@pytest.fixture(autouse=True)
def _no_sleep_and_isolated_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(retry.time, "sleep", lambda _s: None)
    monkeypatch.setattr(storage, "STORAGE_DIR", tmp_path / "batches")
    monkeypatch.setattr(storage, "INPUTS_DIR", tmp_path / "inputs")


@pytest.mark.parametrize("exc", [
    openai.APIConnectionError(request=_REQ),
    openai.APITimeoutError(request=_REQ),
    _status_error(429),
    _status_error(503),
    httpx.ConnectError("[SSL: UNEXPECTED_EOF_WHILE_READING]"),
    ssl.SSLError("UNEXPECTED_EOF_WHILE_READING"),
    ConnectionResetError(),
    genai_errors.APIError(429, {"error": {"message": "quota", "status": "RESOURCE_EXHAUSTED"}}),
])
def test_transient_errors(exc: Exception) -> None:
    assert is_transient(exc)


@pytest.mark.parametrize("exc", [
    _status_error(400),
    _status_error(401),
    _status_error(404),
    ValueError("bad"),
    genai_errors.APIError(400, {"error": {"message": "invalid"}}),
])
def test_permanent_errors(exc: Exception) -> None:
    assert not is_transient(exc)


def test_transient_cause_is_found_through_wrapping() -> None:
    try:
        try:
            raise ssl.SSLError("EOF")
        except ssl.SSLError as inner:
            raise RuntimeError("wrapped") from inner
    except RuntimeError as outer:
        assert is_transient(outer)


def test_with_retries_recovers_and_gives_up() -> None:
    calls = iter([ConnectionResetError(), ConnectionResetError(), "ok"])

    def flaky():
        v = next(calls)
        if isinstance(v, Exception):
            raise v
        return v

    assert with_retries(flaky, what="t") == "ok"

    always = MagicMock(side_effect=ConnectionResetError())
    with pytest.raises(ConnectionResetError):
        with_retries(always, what="t", attempts=3)
    assert always.call_count == 3


def test_with_retries_does_not_retry_permanent_errors() -> None:
    fn = MagicMock(side_effect=_status_error(400))
    with pytest.raises(openai.APIStatusError):
        with_retries(fn, what="t")
    assert fn.call_count == 1


def _patch_provider(monkeypatch: pytest.MonkeyPatch, provider: MagicMock) -> None:
    monkeypatch.setattr(orchestrator, "get_provider", lambda *_a, **_k: provider)
    monkeypatch.setattr(orchestrator, "resolve_api_key", lambda *_a, **_k: "fake")


def test_submit_retries_rate_limit_but_not_network_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = MagicMock()
    provider.submit.side_effect = [_status_error(429), "batch-1"]
    _patch_provider(monkeypatch, provider)
    items = [PromptItem(custom_id="a", prompt="p")]
    result = orchestrator.submit_items(items, "openai", "m", use_cache=False, original_file_path="/tmp/x.json")
    assert result.meta.batch_id == "batch-1"

    # The batch may already exist after a dropped connection: resending could submit it twice
    provider.submit.side_effect = [openai.APIConnectionError(request=_REQ), "batch-2"]
    with pytest.raises(openai.APIConnectionError):
        orchestrator.submit_items(items, "openai", "m", use_cache=False, original_file_path="/tmp/x.json")
    assert is_rate_limited(_status_error(429)) and not is_rate_limited(openai.APIConnectionError(request=_REQ))


def _save_meta() -> BatchMetadata:
    meta = BatchMetadata(batch_id="b1", provider="alibaba", model="m", original_file_path="", file_type="json",
                         source="memory")
    storage.save_batch(meta)
    return meta


def test_fetch_reports_failed_batch_with_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    meta = _save_meta()
    provider = MagicMock()
    provider.check_status.side_effect = [openai.APITimeoutError(request=_REQ), "failed"]
    provider.batch_error_message.return_value = "model 'm' is not supported by the Batch API"
    _patch_provider(monkeypatch, provider)

    result = BatchJob(meta, api_key="fake").fetch()
    assert not result.done and result.failed and result.status == "failed"
    assert result.error == "model 'm' is not supported by the Batch API"
    with pytest.raises(BatchFailedError, match="not supported") as info:
        result.raise_for_failure()
    assert info.value.status == "failed" and info.value.batch_id == "b1"


def test_fetch_running_batch_is_not_failed_and_skips_error_lookup(monkeypatch: pytest.MonkeyPatch) -> None:
    meta = _save_meta()
    provider = MagicMock()
    provider.check_status.return_value = "in_progress"
    _patch_provider(monkeypatch, provider)

    result = BatchJob(meta, api_key="fake").fetch()
    assert not result.done and not result.failed and result.error is None
    result.raise_for_failure()
    provider.batch_error_message.assert_not_called()


def test_openai_batch_error_message_lists_each_error_once() -> None:
    provider = OpenAIBatchProvider(api_key="fake")
    provider.client = MagicMock()
    msg = "The provided model 'qwen3.7-flash' is not supported by the Batch API."
    batch = MagicMock()
    batch.errors.data = [{"code": "model_not_found", "line": i, "message": msg} for i in range(20)]
    provider.client.batches.retrieve.return_value = batch
    assert provider.batch_error_message("b") == msg


def test_batch_fetch_result_without_job_status_change() -> None:
    meta = BatchMetadata(batch_id="b", provider="openai", model="m", original_file_path="", file_type="json",
                         status="completed")
    result = BatchFetchResult(job=BatchJob(meta), done=True)
    assert not result.failed and result.error is None
