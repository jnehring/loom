"""Pydantic models for Loom."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import BaseModel, Field


FileType = Literal["json", "csv", "parquet"]
ProviderName = Literal["openai", "anthropic", "google", "openrouter"]
BatchStatus = Literal[
    "validating", "in_progress", "completed", "failed", "expired", "cancelled", "unknown"
]


class PromptItem(BaseModel):
    """A single unit of work to send to an LLM."""

    custom_id: str
    prompt: str


class BatchMetadata(BaseModel):
    """Persisted record of a submitted batch."""

    batch_id: str
    provider: ProviderName
    model: str
    original_file_path: str
    file_type: FileType
    prompt_column: Optional[str] = None
    # custom_id -> original row index (CSV/Parquet) or original "id" field (JSON)
    id_map: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: BatchStatus = "validating"
    output_path: Optional[str] = None
    with_meta: bool = False
