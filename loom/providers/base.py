from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

from ..core.models import PromptItem


Status = Literal["pending", "in_progress", "completed", "failed", "cancelled", "expired"]


@dataclass
class BatchStatus:
    state: Status
    raw: str  # provider-native status string, for display
    total: int | None = None
    completed: int | None = None


@dataclass
class BatchResult:
    """custom_id -> llm response text. Errored items map to None."""
    responses: dict[str, str | None]


class BatchProvider(ABC):
    """Strategy interface for a batch-capable LLM provider."""

    name: str  # "openai" | "anthropic" | "google"
    env_var: str  # name of env var holding the API key

    def __init__(self, api_key: str | None = None):
        import os
        self.api_key = api_key or os.environ.get(self.env_var)
        if not self.api_key:
            raise RuntimeError(
                f"No API key for {self.name}. Set {self.env_var} or pass --api-key."
            )

    @abstractmethod
    def submit(self, prompts: list[PromptItem], model: str) -> str:
        """Submit a batch and return the provider's batch id."""

    @abstractmethod
    def status(self, batch_id: str) -> BatchStatus:
        """Poll for current status."""

    @abstractmethod
    def fetch_results(self, batch_id: str) -> BatchResult:
        """Download results. Only valid when status.state == 'completed'."""
