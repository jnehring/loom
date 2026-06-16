"""Google Gemini Batch API provider.

Docs: https://ai.google.dev/gemini-api/docs/batch-api

Uses the google-genai SDK with inline requests. For very large inputs the
SDK supports a file/GCS upload path which is out of scope for v1.
"""

from __future__ import annotations

import json
from typing import Any

from ..core.models import BatchStatus, PromptItem
from ..utils.errors import format_api_error
from .base import BatchProvider


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

        # Reconstruct the list of custom IDs in submission order
        custom_ids = []
        if id_map:
            custom_ids = list(id_map.keys())
            try:
                custom_ids.sort(key=lambda k: int(id_map[k]))
            except ValueError:
                pass

        if inlined:
            for i, entry in enumerate(inlined):
                meta = getattr(entry, "metadata", None) or {}
                cid = meta.get("custom_id", "") if isinstance(meta, dict) else ""
                if not cid and i < len(custom_ids):
                    cid = custom_ids[i]
                if not cid:
                    cid = f"idx-{i}"

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
            for i, line in enumerate(lines):
                obj = json.loads(line)
                cid = (obj.get("metadata") or {}).get("custom_id", "")
                if not cid and i < len(custom_ids):
                    cid = custom_ids[i]
                if not cid:
                    cid = f"idx-{i}"

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
