"""OpenAI Batch API provider.

Docs: https://developers.openai.com/api/docs/guides/batch
"""

from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

from ..core.models import BatchStatus, PromptItem
from .base import BatchProvider

if TYPE_CHECKING:
    from openai import OpenAI  # noqa: F401


_STATUS_MAP = {
    "validating": "validating",
    "in_progress": "in_progress",
    "finalizing": "in_progress",
    "completed": "completed",
    "failed": "failed",
    "expired": "expired",
    "cancelling": "in_progress",
    "cancelled": "cancelled",
}


class OpenAIBatchProvider(BatchProvider):
    name = "openai"

    def __init__(self, api_key: str) -> None:
        super().__init__(api_key)
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def _build_jsonl(self, items: list[PromptItem], model: str) -> bytes:
        buf = io.BytesIO()
        for it in items:
            line = {
                "custom_id": it.custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [{"role": "user", "content": it.prompt}],
                },
            }
            buf.write((json.dumps(line) + "\n").encode("utf-8"))
        buf.seek(0)
        return buf.getvalue()

    def submit(self, items: list[PromptItem], model: str) -> str:
        data = self._build_jsonl(items, model)
        f = self.client.files.create(
            file=("loom_batch.jsonl", data),
            purpose="batch",
        )
        batch = self.client.batches.create(
            input_file_id=f.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        return batch.id

    def check_status(self, batch_id: str) -> BatchStatus:
        batch = self.client.batches.retrieve(batch_id)
        return _STATUS_MAP.get(batch.status, "unknown")  # type: ignore[return-value]

    def download_results(self, batch_id: str) -> dict[str, str]:
        batch = self.client.batches.retrieve(batch_id)
        if not batch.output_file_id:
            return {}
        content = self.client.files.content(batch.output_file_id)
        text = content.read().decode("utf-8") if hasattr(content, "read") else content.text
        out: dict[str, str] = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj.get("custom_id", "")
            resp = obj.get("response", {}) or {}
            body = resp.get("body", {}) or {}
            choices = body.get("choices") or []
            if choices:
                msg = choices[0].get("message", {}) or {}
                out[cid] = msg.get("content", "") or ""
            else:
                out[cid] = ""
        return out
