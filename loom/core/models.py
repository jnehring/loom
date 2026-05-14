from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from pydantic import BaseModel, Field


FileType = Literal["json", "csv"]
ProviderName = Literal["openai", "anthropic", "google"]


class PromptItem(BaseModel):
    """A single prompt to send to an LLM."""
    id: str
    prompt: str


class BatchMetadata(BaseModel):
    """Persisted state for a submitted batch. Stored at ~/.loom/batches/{provider}_{batch_id}.json"""
    batch_id: str
    provider: ProviderName
    model: str
    original_file_path: str
    file_type: FileType
    prompt_column: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    # Maps the custom_id we send to the provider back to the original row identity.
    # For JSON: custom_id -> original "id" field.
    # For CSV: custom_id -> original row index (as string).
    id_map: dict[str, str] = Field(default_factory=dict)
