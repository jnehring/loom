from __future__ import annotations

from openai import OpenAI

from ..core.models import PromptItem
from ..utils.converters import dump_jsonl, parse_jsonl
from .base import BatchProvider, BatchResult, BatchStatus, Status


_STATUS_MAP: dict[str, Status] = {
    "validating": "pending",
    "in_progress": "in_progress",
    "finalizing": "in_progress",
    "completed": "completed",
    "failed": "failed",
    "expired": "expired",
    "cancelling": "in_progress",
    "cancelled": "cancelled",
}


class OpenAIProvider(BatchProvider):
    name = "openai"
    env_var = "OPENAI_API_KEY"

    def _client(self) -> OpenAI:
        return OpenAI(api_key=self.api_key)

    def submit(self, prompts: list[PromptItem], model: str) -> str:
        requests = [
            {
                "custom_id": p.id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": [{"role": "user", "content": p.prompt}],
                },
            }
            for p in prompts
        ]
        client = self._client()
        file_obj = client.files.create(
            file=("batch.jsonl", dump_jsonl(requests)),
            purpose="batch",
        )
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        return batch.id

    def status(self, batch_id: str) -> BatchStatus:
        batch = self._client().batches.retrieve(batch_id)
        counts = batch.request_counts
        return BatchStatus(
            state=_STATUS_MAP.get(batch.status, "pending"),
            raw=batch.status,
            total=counts.total if counts else None,
            completed=counts.completed if counts else None,
        )

    def fetch_results(self, batch_id: str) -> BatchResult:
        client = self._client()
        batch = client.batches.retrieve(batch_id)
        if not batch.output_file_id:
            raise RuntimeError(f"Batch {batch_id} has no output_file_id (status={batch.status})")
        content = client.files.content(batch.output_file_id)
        lines = parse_jsonl(content.text)

        responses: dict[str, str | None] = {}
        for line in lines:
            cid = line.get("custom_id")
            if not cid:
                continue
            if line.get("error"):
                responses[cid] = None
                continue
            try:
                responses[cid] = line["response"]["body"]["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                responses[cid] = None
        return BatchResult(responses=responses)
