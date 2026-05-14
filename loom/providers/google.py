from __future__ import annotations

from google import genai
from google.genai import types

from ..core.models import PromptItem
from .base import BatchProvider, BatchResult, BatchStatus, Status


_STATUS_MAP: dict[str, Status] = {
    "JOB_STATE_UNSPECIFIED": "pending",
    "JOB_STATE_QUEUED": "pending",
    "JOB_STATE_PENDING": "pending",
    "JOB_STATE_RUNNING": "in_progress",
    "JOB_STATE_UPDATING": "in_progress",
    "JOB_STATE_CANCELLING": "in_progress",
    "JOB_STATE_PAUSED": "pending",
    "JOB_STATE_SUCCEEDED": "completed",
    "JOB_STATE_PARTIALLY_SUCCEEDED": "completed",
    "JOB_STATE_FAILED": "failed",
    "JOB_STATE_CANCELLED": "cancelled",
    "JOB_STATE_EXPIRED": "expired",
}


class GoogleProvider(BatchProvider):
    name = "google"
    env_var = "GOOGLE_API_KEY"

    _cached_client: genai.Client | None = None

    def _client(self) -> genai.Client:
        # Cache the Client: genai.Client.__del__ closes its httpx session, so a
        # short-lived temporary will be garbage-collected mid-request.
        if self._cached_client is None:
            self._cached_client = genai.Client(api_key=self.api_key)
        return self._cached_client

    def submit(self, prompts: list[PromptItem], model: str) -> str:
        requests = [
            types.InlinedRequest(
                contents=[{"role": "user", "parts": [{"text": p.prompt}]}],
                metadata={"custom_id": p.id},
            )
            for p in prompts
        ]
        job = self._client().batches.create(
            model=model,
            src=requests,
            config={"display_name": "loom-batch"},
        )
        if not job.name:
            raise RuntimeError("Gemini batch creation returned no name")
        # job.name is "batches/abc..." — strip the prefix so storage keys match other providers.
        return job.name.split("/", 1)[-1]

    def _full_name(self, batch_id: str) -> str:
        return batch_id if batch_id.startswith("batches/") else f"batches/{batch_id}"

    def status(self, batch_id: str) -> BatchStatus:
        job = self._client().batches.get(name=self._full_name(batch_id))
        state_str = job.state.value if job.state else ""
        total = len(job.dest.inlined_responses) if (job.dest and job.dest.inlined_responses) else None
        return BatchStatus(
            state=_STATUS_MAP.get(state_str, "pending"),
            raw=state_str,
            total=total,
            completed=total,
        )

    def fetch_results(self, batch_id: str) -> BatchResult:
        job = self._client().batches.get(name=self._full_name(batch_id))
        if not job.dest or not job.dest.inlined_responses:
            raise RuntimeError(f"Batch {batch_id} has no inlined_responses (state={job.state})")

        responses: dict[str, str | None] = {}
        for item in job.dest.inlined_responses:
            cid = (item.metadata or {}).get("custom_id")
            if not cid:
                continue
            if item.error or not item.response:
                responses[cid] = None
                continue
            try:
                responses[cid] = item.response.text or ""
            except Exception:
                responses[cid] = None
        return BatchResult(responses=responses)
