"""Abstract base class for synchronous (non-batch) providers.

A SyncProvider is used by the ``loom run --sync`` flow. It makes one
chat-completion call per prompt and returns the text response.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class SyncProvider(ABC):
    name: str

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    @abstractmethod
    def generate(self, prompt: str, model: str) -> str:
        """Return the assistant's text reply for a single prompt."""

    def count_tokens(self, prompt: str, model: str) -> Optional[int]:
        """Return the input-token count for a single prompt, or None if the
        provider does not expose a token-counting API."""
        return None
