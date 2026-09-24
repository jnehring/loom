"""Translate :class:`GenerationParams` into each provider's request fields.

One place per provider, shared by the batch and the sync implementation, so both
send exactly the same settings. Unsupported settings raise
:class:`~loom.utils.errors.UnsupportedParameterError` instead of being dropped.

| Setting             | OpenAI                  | OpenRouter          | Anthropic        | Google                 |
|---------------------|-------------------------|---------------------|------------------|------------------------|
| temperature         | temperature             | temperature         | temperature      | temperature            |
| max_tokens          | max_completion_tokens   | max_tokens          | max_tokens       | max_output_tokens      |
| top_p               | top_p                   | top_p               | top_p            | top_p                  |
| top_k               | –                       | top_k               | top_k            | top_k                  |
| stop                | stop                    | stop                | stop_sequences   | stop_sequences         |
| seed                | seed                    | seed                | –                | seed                   |
| presence_penalty    | presence_penalty        | presence_penalty    | –                | presence_penalty       |
| frequency_penalty   | frequency_penalty       | frequency_penalty   | –                | frequency_penalty      |
| system              | system message          | system message      | system           | system_instruction     |
| json_mode           | response_format         | response_format     | –                | response_mime_type     |
"""

from __future__ import annotations

from typing import Any, Optional

from ..core.models import GenerationParams
from ..utils.errors import UnsupportedParameterError

ANTHROPIC_DEFAULT_MAX_TOKENS = 4096

_UNSUPPORTED = {
    "openai": {"top_k"},
    "openrouter": set(),
    "anthropic": {"seed", "presence_penalty", "frequency_penalty", "json_mode"},
    "google": set(),
}


def check_supported(provider: str, params: Optional[GenerationParams]) -> GenerationParams:
    """Return ``params`` (or defaults); raise if a set field is unsupported by ``provider``."""
    params = params or GenerationParams()
    bad = [name for name in _UNSUPPORTED.get(provider, set()) if name in params.set_fields()]
    if bad:
        raise UnsupportedParameterError(provider, bad)
    return params


# ---------- OpenAI-compatible (OpenAI, OpenRouter) ----------

def openai_messages(prompt: str, params: GenerationParams) -> list[dict[str, str]]:
    messages = []
    if params.system is not None:
        messages.append({"role": "system", "content": params.system})
    messages.append({"role": "user", "content": prompt})
    return messages


def openai_body(params: GenerationParams, *, provider: str = "openai") -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(fields, extra_body)`` for chat completions.

    ``fields`` are standard chat-completion fields. ``extra_body`` holds
    non-standard fields (OpenRouter ``top_k`` and everything in ``params.extra``);
    batch requests merge both into the JSONL body, the SDK passes the second via
    ``extra_body``.
    """
    params = check_supported(provider, params)
    fields: dict[str, Any] = {}
    if params.temperature is not None:
        fields["temperature"] = params.temperature
    if params.max_tokens is not None:
        fields["max_completion_tokens" if provider == "openai" else "max_tokens"] = params.max_tokens
    if params.top_p is not None:
        fields["top_p"] = params.top_p
    if params.stop is not None:
        fields["stop"] = list(params.stop)
    if params.seed is not None:
        fields["seed"] = params.seed
    if params.presence_penalty is not None:
        fields["presence_penalty"] = params.presence_penalty
    if params.frequency_penalty is not None:
        fields["frequency_penalty"] = params.frequency_penalty
    if params.json_mode:
        fields["response_format"] = {"type": "json_object"}
    extra: dict[str, Any] = {}
    if params.top_k is not None:
        extra["top_k"] = params.top_k
    extra.update(params.extra)
    return fields, extra


# ---------- Anthropic ----------

def anthropic_params(params: GenerationParams) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(fields, extra)`` for ``messages.create`` (``max_tokens`` is always set)."""
    params = check_supported("anthropic", params)
    fields: dict[str, Any] = {"max_tokens": params.max_tokens or ANTHROPIC_DEFAULT_MAX_TOKENS}
    if params.temperature is not None:
        fields["temperature"] = params.temperature
    if params.top_p is not None:
        fields["top_p"] = params.top_p
    if params.top_k is not None:
        fields["top_k"] = params.top_k
    if params.stop is not None:
        fields["stop_sequences"] = list(params.stop)
    if params.system is not None:
        fields["system"] = params.system
    return fields, dict(params.extra)


# ---------- Google ----------

def google_config(params: GenerationParams) -> dict[str, Any]:
    """``GenerateContentConfig`` fields as a plain dict (used inline in batch requests and for sync calls)."""
    params = check_supported("google", params)
    config: dict[str, Any] = {}
    if params.temperature is not None:
        config["temperature"] = params.temperature
    if params.max_tokens is not None:
        config["max_output_tokens"] = params.max_tokens
    if params.top_p is not None:
        config["top_p"] = params.top_p
    if params.top_k is not None:
        config["top_k"] = params.top_k
    if params.stop is not None:
        config["stop_sequences"] = list(params.stop)
    if params.seed is not None:
        config["seed"] = params.seed
    if params.presence_penalty is not None:
        config["presence_penalty"] = params.presence_penalty
    if params.frequency_penalty is not None:
        config["frequency_penalty"] = params.frequency_penalty
    if params.system is not None:
        config["system_instruction"] = {"parts": [{"text": params.system}]}
    if params.json_mode:
        config["response_mime_type"] = "application/json"
    config.update(params.extra)
    return config
