"""Pydantic models for Loom."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


FileType = Literal["json", "csv", "parquet"]
ProviderName = Literal["openai", "anthropic", "google", "openrouter"]
BatchStatus = Literal[
    "validating", "in_progress", "completed", "failed", "expired", "cancelled", "unknown"
]
BatchSource = Literal["file", "memory"]


class GenerationParams(BaseModel):
    """Sampling and output settings, translated to each provider's request format.

    Every field is optional; ``None`` means "use the provider default". Loom never
    silently drops a setting: a field a provider does not support raises
    :class:`~loom.utils.errors.UnsupportedParameterError` when the request is built.

    ``extra`` is passed through unchanged into the provider request (OpenAI and
    OpenRouter: request body; Anthropic: message params; Google: ``GenerateContentConfig``)
    for provider-specific options Loom does not model, e.g. ``{"reasoning_effort": "low"}``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    temperature: Optional[float] = Field(None, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, gt=0, description="Maximum number of output tokens.")
    top_p: Optional[float] = Field(None, gt=0, le=1)
    top_k: Optional[int] = Field(None, gt=0)
    stop: Optional[list[str]] = Field(None, description="Stop sequences.")
    seed: Optional[int] = None
    presence_penalty: Optional[float] = Field(None, ge=-2, le=2)
    frequency_penalty: Optional[float] = Field(None, ge=-2, le=2)
    system: Optional[str] = Field(None, description="System prompt / instruction.")
    json_mode: bool = Field(False, description="Ask the model for a JSON object as the response.")
    extra: dict[str, Any] = Field(default_factory=dict)

    def is_default(self) -> bool:
        return not self.set_fields()

    def set_fields(self) -> dict[str, Any]:
        """Fields that differ from the provider default, in a stable order."""
        data = self.model_dump()
        out = {k: v for k, v in data.items() if v is not None and k not in ("json_mode", "extra")}
        if self.json_mode:
            out["json_mode"] = True
        if self.extra:
            out["extra"] = self.extra
        return out

    def cache_token(self) -> str:
        """Canonical JSON of the non-default settings ('' when all defaults).

        Part of the response-cache key, so answers generated with different
        settings are cached separately while default-setting keys stay unchanged.
        """
        fields = self.set_fields()
        return json.dumps(fields, sort_keys=True, ensure_ascii=False, separators=(",", ":")) if fields else ""

    def merged(self, override: Optional["GenerationParams"]) -> "GenerationParams":
        """``override``'s explicitly set fields on top of these settings."""
        if override is None:
            return self
        data = self.model_dump()
        data.update(override.model_dump(exclude_unset=True))
        return GenerationParams(**data)


def as_params(value: "GenerationParams | dict[str, Any] | None") -> GenerationParams:
    """Accept a :class:`GenerationParams`, a plain dict, or ``None``."""
    if value is None:
        return GenerationParams()
    if isinstance(value, GenerationParams):
        return value
    return GenerationParams(**value)


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
    # Responses served from cache at submit time (custom_id -> response text).
    cached_responses: dict[str, str] = Field(default_factory=dict)
    # Cache directory used at submit time, so fetch writes back to the same place.
    cache_dir: Optional[str] = None
    # Whether the batch was submitted from a file or an in-memory prompt list.
    source: BatchSource = "file"
    # Generation settings used at submit time (fetch writes the cache under the same key).
    params: GenerationParams = Field(default_factory=GenerationParams)
