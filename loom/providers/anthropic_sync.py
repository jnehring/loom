"""Anthropic synchronous messages provider."""

from __future__ import annotations

from typing import Optional

from ..core.models import GenerationParams
from .params import anthropic_params
from .sync_base import SyncProvider


class AnthropicSyncProvider(SyncProvider):
    name = "anthropic"

    def __init__(self, api_key: str) -> None:
        super().__init__(api_key)
        from anthropic import Anthropic
        self.client = Anthropic(api_key=api_key)

    def generate(self, prompt: str, model: str, params: Optional[GenerationParams] = None) -> str:
        fields, extra = anthropic_params(params or GenerationParams())
        msg = self.client.messages.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            **fields,
            **({"extra_body": extra} if extra else {}),
        )
        parts = []
        for block in msg.content or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts)

    def count_tokens(self, prompt: str, model: str) -> Optional[int]:
        resp = self.client.messages.count_tokens(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return int(getattr(resp, "input_tokens", 0) or 0)
