"""OpenAI synchronous chat-completions provider."""

from __future__ import annotations

from .sync_base import SyncProvider


class OpenAISyncProvider(SyncProvider):
    name = "openai"

    def __init__(self, api_key: str) -> None:
        super().__init__(api_key)
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt: str, model: str) -> str:
        resp = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        choices = resp.choices or []
        if not choices:
            return ""
        return choices[0].message.content or ""
