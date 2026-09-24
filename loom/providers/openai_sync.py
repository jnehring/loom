"""OpenAI synchronous chat-completions provider."""

from __future__ import annotations

from typing import Optional

from ..core.models import GenerationParams
from .params import openai_body, openai_messages
from .sync_base import SyncProvider


class OpenAISyncProvider(SyncProvider):
    """OpenAI chat completions; subclasses reuse it for OpenAI-compatible endpoints (``base_url``)."""

    name = "openai"
    base_url: Optional[str] = None

    def __init__(self, api_key: str) -> None:
        super().__init__(api_key)
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, **({"base_url": self.base_url} if self.base_url else {}))

    def generate(self, prompt: str, model: str, params: Optional[GenerationParams] = None) -> str:
        params = params or GenerationParams()
        fields, extra = openai_body(params, provider=self.name)
        resp = self.client.chat.completions.create(
            model=model,
            messages=openai_messages(prompt, params),
            **fields,
            **({"extra_body": extra} if extra else {}),
        )
        choices = resp.choices or []
        if not choices:
            return ""
        return choices[0].message.content or ""

    def count_tokens(self, prompt: str, model: str) -> Optional[int]:
        resp = self.client.responses.input_tokens.count(
            model=model,
            input=prompt,
        )
        return int(getattr(resp, "input_tokens", 0) or 0)
