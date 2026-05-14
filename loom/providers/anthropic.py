from __future__ import annotations

import httpx

from ..core.models import PromptItem
from ..utils.converters import parse_jsonl
from .base import BatchProvider, BatchResult, BatchStatus, Status


BASE_URL = "https://api.anthropic.com/v1"
API_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096


_STATUS_MAP: dict[str, Status] = {
    "in_progress": "in_progress",
    "canceling": "in_progress",
    "ended": "completed",  # canonical "ended" — individual requests may still have failed
}


class AnthropicProvider(BatchProvider):
    name = "anthropic"
    env_var = "ANTHROPIC_API_KEY"

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self.api_key or "",
            "anthropic-version": API_VERSION,
            "content-type": "application/json",
        }

    def _client(self) -> httpx.Client:
        return httpx.Client(base_url=BASE_URL, headers=self._headers(), timeout=120.0)

    def submit(self, prompts: list[PromptItem], model: str) -> str:
        requests = [
            {
                "custom_id": p.id,
                "params": {
                    "model": model,
                    "max_tokens": DEFAULT_MAX_TOKENS,
                    "messages": [{"role": "user", "content": p.prompt}],
                },
            }
            for p in prompts
        ]
        with self._client() as client:
            r = client.post("/messages/batches", json={"requests": requests})
            r.raise_for_status()
            return r.json()["id"]

    def status(self, batch_id: str) -> BatchStatus:
        with self._client() as client:
            r = client.get(f"/messages/batches/{batch_id}")
            r.raise_for_status()
            body = r.json()
        counts = body.get("request_counts") or {}
        total = sum(counts.values()) if counts else None
        completed = (counts.get("succeeded", 0) + counts.get("errored", 0)
                     + counts.get("canceled", 0) + counts.get("expired", 0)) if counts else None
        return BatchStatus(
            state=_STATUS_MAP.get(body.get("processing_status", ""), "pending"),
            raw=body.get("processing_status", ""),
            total=total,
            completed=completed,
        )

    def fetch_results(self, batch_id: str) -> BatchResult:
        with self._client() as client:
            r = client.get(f"/messages/batches/{batch_id}")
            r.raise_for_status()
            results_url = r.json().get("results_url")
            if not results_url:
                raise RuntimeError(f"Batch {batch_id} has no results_url yet")
            # results_url is a fully-qualified URL; use a separate request with auth headers.
            r = client.get(results_url)
            r.raise_for_status()
            lines = parse_jsonl(r.text)

        responses: dict[str, str | None] = {}
        for line in lines:
            cid = line.get("custom_id")
            if not cid:
                continue
            result = line.get("result") or {}
            if result.get("type") != "succeeded":
                responses[cid] = None
                continue
            try:
                blocks = result["message"]["content"]
                text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                responses[cid] = text
            except (KeyError, TypeError):
                responses[cid] = None
        return BatchResult(responses=responses)
