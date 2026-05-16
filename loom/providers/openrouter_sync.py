"""OpenRouter synchronous provider.

OpenRouter (https://openrouter.ai) exposes an OpenAI-compatible chat-completions
API at ``https://openrouter.ai/api/v1``. It does NOT offer a batch API, so it
is only available via ``loom run --sync``.
"""

from __future__ import annotations

from .sync_base import SyncProvider


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterSyncProvider(SyncProvider):
    name = "openrouter"

    def __init__(self, api_key: str) -> None:
        super().__init__(api_key)
        from openai import OpenAI
        self.client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    def generate(self, prompt: str, model: str) -> str:
        resp = self.client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        choices = resp.choices or []
        if not choices:
            return ""
        return choices[0].message.content or ""
