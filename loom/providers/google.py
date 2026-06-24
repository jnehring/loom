"""Google Gemini Batch API provider.

Docs: https://ai.google.dev/gemini-api/docs/batch-api

Uses the google-genai SDK with inline requests. For very large inputs the
SDK supports a file/GCS upload path which is out of scope for v1.

Google's batch API may return inlined responses out of submission order
(especially for 100+ requests). Results are matched via ``metadata.custom_id``
on each response — never by list position.
"""

from __future__ import annotations

import json
from typing import Any

from ..core.models import BatchStatus, PromptItem
from ..utils.errors import format_api_error
from .base import BatchProvider


_METADATA_PATCHED = False


def _ensure_inlined_response_metadata() -> None:
    """Backfill metadata on InlinedResponse for google-genai < 1.61.0."""
    global _METADATA_PATCHED
    if _METADATA_PATCHED:
        return
    _METADATA_PATCHED = True

    from google.genai import types

    if "metadata" in types.InlinedResponse.model_fields:
        return

    import typing

    import google.genai.batches as batches
    from google.genai._common import get_value_by_path as getv
    from google.genai._common import set_value_by_path as setv
    from pydantic.fields import FieldInfo

    original = batches._InlinedResponse_from_mldev

    def patched(
        from_object: dict[str, typing.Any] | object,
        parent_object: dict[str, typing.Any] | None = None,
    ) -> dict[str, typing.Any]:
        to_object = original(from_object, parent_object)
        if getv(from_object, ["metadata"]) is not None:
            setv(to_object, ["metadata"], getv(from_object, ["metadata"]))
        return to_object

    batches._InlinedResponse_from_mldev = patched
    types.InlinedResponse.model_fields["metadata"] = FieldInfo(
        annotation=dict[str, Any],
        default_factory=dict,
        description="The metadata associated with the request.",
    )
    types.InlinedResponse.model_rebuild(force=True)
    if hasattr(types, "BatchJobDestination"):
        types.BatchJobDestination.model_rebuild(force=True)
    if hasattr(types, "BatchJob"):
        types.BatchJob.model_rebuild(force=True)


def _metadata_dict(entry: Any) -> dict[str, Any]:
    meta = getattr(entry, "metadata", None)
    if isinstance(meta, dict):
        return meta
    if isinstance(entry, dict):
        raw = entry.get("metadata")
        if isinstance(raw, dict):
            return raw
    return {}


def _custom_id_from_metadata(meta: dict[str, Any]) -> str:
    for key in ("custom_id", "id", "request_id"):
        val = meta.get(key)
        if val is not None and str(val).strip():
            return str(val).strip()
    return ""


def _missing_metadata_error() -> ValueError:
    return ValueError(
        "Google Batch API returned one or more responses without metadata; "
        "cannot reliably match results to prompts because the API may return "
        "responses out of submission order. Upgrade google-genai to >=1.61.0 "
        "and re-fetch. If the problem persists, re-submit the batch."
    )


_STATE_MAP = {
    "JOB_STATE_QUEUED": "validating",
    "JOB_STATE_PENDING": "validating",
    "JOB_STATE_RUNNING": "in_progress",
    "JOB_STATE_SUCCEEDED": "completed",
    "JOB_STATE_FAILED": "failed",
    "JOB_STATE_CANCELLED": "cancelled",
    "JOB_STATE_EXPIRED": "expired",
}


class GoogleBatchProvider(BatchProvider):
    name = "google"

    def __init__(self, api_key: str) -> None:
        super().__init__(api_key)
        _ensure_inlined_response_metadata()
        from google import genai
        self.client = genai.Client(api_key=api_key)

    def submit(self, items: list[PromptItem], model: str) -> str:
        inlined = [
            {
                "contents": [{"parts": [{"text": it.prompt}], "role": "user"}],
                "metadata": {"custom_id": it.custom_id},
            }
            for it in items
        ]
        job = self.client.batches.create(
            model=model,
            src=inlined,
            config={"display_name": "loom-batch"},
        )
        return job.name

    def _retrieve(self, batch_id: str) -> Any:
        return self.client.batches.get(name=batch_id)

    def check_status(self, batch_id: str) -> BatchStatus:
        job = self._retrieve(batch_id)
        state = getattr(job.state, "name", str(job.state))
        return _STATE_MAP.get(state, "unknown")  # type: ignore[return-value]

    def download_results(
        self, batch_id: str, id_map: dict[str, str] | None = None
    ) -> tuple[dict[str, str], dict[str, str]]:
        job = self._retrieve(batch_id)
        out: dict[str, str] = {}
        errors: dict[str, str] = {}
        dest = getattr(job, "dest", None)
        inlined = getattr(dest, "inlined_responses", None) if dest else None

        if inlined:
            for entry in inlined:
                meta = _metadata_dict(entry)
                cid = _custom_id_from_metadata(meta)
                if not cid:
                    raise _missing_metadata_error()

                resp = getattr(entry, "response", None)
                candidates = getattr(resp, "candidates", None) if resp is not None else None
                if candidates:
                    content = getattr(candidates[0], "content", None)
                    parts = getattr(content, "parts", None) or []
                    out[cid] = "".join(getattr(p, "text", "") or "" for p in parts)
                    continue

                entry_error = getattr(entry, "error", None)
                if entry_error:
                    out[cid] = ""
                    msg = format_api_error(entry_error)
                    if msg:
                        errors[cid] = msg
                    continue

                out[cid] = ""
            return out, errors

        # Fallback: file-based output
        file_name = getattr(dest, "file_name", None) if dest else None
        if file_name:
            data = self.client.files.download(file=file_name)
            text = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)
            lines = [line for line in text.splitlines() if line.strip()]
            for line in lines:
                obj = json.loads(line)
                meta = obj.get("metadata") or {}
                if not isinstance(meta, dict):
                    meta = {}
                cid = _custom_id_from_metadata(meta)
                if not cid:
                    raise _missing_metadata_error()

                resp = obj.get("response", {}) or {}
                candidates = resp.get("candidates", []) or []
                if candidates:
                    parts = (candidates[0].get("content", {}) or {}).get("parts", []) or []
                    out[cid] = "".join(p.get("text", "") for p in parts)
                    continue

                if obj.get("error"):
                    out[cid] = ""
                    msg = format_api_error(obj["error"])
                    if msg:
                        errors[cid] = msg
                    continue

                out[cid] = ""
        return out, errors

    def batch_error_message(self, batch_id: str) -> str | None:
        job = self._retrieve(batch_id)
        return format_api_error(getattr(job, "error", None)) or None
