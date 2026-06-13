"""Abstract base class for batch providers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.models import BatchStatus, PromptItem


class BatchProvider(ABC):
    name: str

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    @abstractmethod
    def submit(self, items: list[PromptItem], model: str) -> str:
        """Submit a batch; return the provider's batch_id."""

    @abstractmethod
    def check_status(self, batch_id: str) -> BatchStatus:
        """Return a normalized status string."""

    @abstractmethod
    def download_results(self, batch_id: str, id_map: dict[str, str] | None = None) -> dict[str, str]:
        """Return {custom_id: response_text} for a completed batch."""

