"""Google Gemini synchronous generate-content provider."""

from __future__ import annotations

from typing import Optional

from ..core.models import GenerationParams
from .params import google_config
from .sync_base import SyncProvider


class GoogleSyncProvider(SyncProvider):
    name = "google"

    def __init__(self, api_key: str) -> None:
        super().__init__(api_key)
        from google import genai
        self.client = genai.Client(api_key=api_key)

    def generate(self, prompt: str, model: str, params: Optional[GenerationParams] = None) -> str:
        config = google_config(params or GenerationParams())
        resp = self.client.models.generate_content(model=model, contents=prompt, **({"config": config} if config else {}))
        text = getattr(resp, "text", None)
        if text:
            return text
        # Fallback: walk candidates -> content.parts -> text
        parts_out = []
        for cand in getattr(resp, "candidates", None) or []:
            content = getattr(cand, "content", None)
            for p in getattr(content, "parts", None) or []:
                t = getattr(p, "text", None)
                if t:
                    parts_out.append(t)
        return "".join(parts_out)

    def count_tokens(self, prompt: str, model: str) -> Optional[int]:
        resp = self.client.models.count_tokens(model=model, contents=prompt)
        total = getattr(resp, "total_tokens", None)
        return int(total) if total is not None else None
