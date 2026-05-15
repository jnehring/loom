"""Google Gemini Batch API provider.

Docs: https://ai.google.dev/gemini-api/docs/batch-api

Uses the google-genai SDK with inline requests. For very large inputs the
SDK supports a file/GCS upload path which is out of scope for v1.
"""

from __future__ import annotations

import json
from typing import Any

from ..core.models import BatchStatus, PromptItem
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

    def download_results(self, batch_id: str) -> dict[str, str]:
        job = self._retrieve(batch_id)
        out: dict[str, str] = {}
        dest = getattr(job, "dest", None)
        inlined = getattr(dest, "inlined_responses", None) if dest else None
        if inlined:
            for entry in inlined:
                meta = getattr(entry, "metadata", None) or {}
                cid = meta.get("custom_id", "") if isinstance(meta, dict) else ""
                resp = getattr(entry, "response", None)
                text = ""
                if resp is not None:
                    candidates = getattr(resp, "candidates", None) or []
                    if candidates:
                        content = getattr(candidates[0], "content", None)
                        parts = getattr(content, "parts", None) or []
                        text = "".join(getattr(p, "text", "") or "" for p in parts)
                out[cid] = text
            return out
        # Fallback: file-based output
        file_name = getattr(dest, "file_name", None) if dest else None
        if file_name:
            data = self.client.files.download(file=file_name)
            text = data.decode("utf-8") if isinstance(data, (bytes, bytearray)) else str(data)
            for line in text.splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                cid = (obj.get("metadata") or {}).get("custom_id", "")
                resp = obj.get("response", {}) or {}
                candidates = resp.get("candidates", []) or []
                if candidates:
                    parts = (candidates[0].get("content", {}) or {}).get("parts", []) or []
                    out[cid] = "".join(p.get("text", "") for p in parts)
                else:
                    out[cid] = ""
        return out
