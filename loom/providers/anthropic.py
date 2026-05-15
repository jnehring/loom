"""Anthropic Message Batches provider.

Docs: https://platform.claude.com/docs/en/build-with-claude/batch-processing
"""

from __future__ import annotations

from ..core.models import BatchStatus, PromptItem
from .base import BatchProvider


_STATUS_MAP = {
    "in_progress": "in_progress",
    "canceling": "in_progress",
    "ended": "completed",  # individual results may still be failed
}


class AnthropicBatchProvider(BatchProvider):
    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        super().__init__(api_key)
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)

    def submit(self, items: list[PromptItem], model: str) -> str:
        from anthropic.types.messages.batch_create_params import Request
        from anthropic.types.message_create_params import MessageCreateParamsNonStreaming

        requests = [
            Request(
                custom_id=it.custom_id,
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": it.prompt}],
                ),
            )
            for it in items
        ]
        batch = self.client.messages.batches.create(requests=requests)
        return batch.id

    def check_status(self, batch_id: str) -> BatchStatus:
        batch = self.client.messages.batches.retrieve(batch_id)
        return _STATUS_MAP.get(batch.processing_status, "unknown")  # type: ignore[return-value]

    def download_results(self, batch_id: str) -> dict[str, str]:
        out: dict[str, str] = {}
        for result in self.client.messages.batches.results(batch_id):
            cid = result.custom_id
            if result.result.type == "succeeded":
                message = result.result.message
                # Concatenate text blocks
                parts = []
                for block in message.content:
                    if getattr(block, "type", None) == "text":
                        parts.append(block.text)
                out[cid] = "".join(parts)
            else:
                out[cid] = ""
        return out
