"""Helpers for formatting provider / prompt errors."""

from __future__ import annotations

from typing import Any


def format_exception(exc: Exception) -> str:
    msg = str(exc).strip()
    return msg if msg else type(exc).__name__


def format_api_error(obj: Any) -> str:
    """Best-effort message extraction from provider error payloads."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, dict):
        for key in ("message", "error", "detail", "status"):
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return str(obj)
    msg = getattr(obj, "message", None)
    if isinstance(msg, str) and msg.strip():
        return msg.strip()
    return str(obj).strip()


def summarize_messages(messages: list[str]) -> str | None:
    """Join the distinct error messages in order (a failed batch repeats the same error once per input line)."""
    distinct = list(dict.fromkeys(m for m in messages if m))
    return "; ".join(distinct) if distinct else None


class BatchFailedError(RuntimeError):
    """A batch ended without results (provider status failed, expired or cancelled)."""

    def __init__(self, batch_id: str, status: str, message: str | None = None) -> None:
        super().__init__(f"Batch {batch_id} {status}" + (f": {message}" if message else ""))
        self.batch_id = batch_id
        self.status = status
        self.message = message


class UnsupportedParameterError(ValueError):
    """A generation setting the chosen provider does not support."""

    def __init__(self, provider: str, names: list[str]) -> None:
        joined = ", ".join(sorted(names))
        super().__init__(
            f"Provider '{provider}' does not support: {joined}. "
            "Remove the setting or pass a provider-specific option via 'extra'."
        )
        self.provider = provider
        self.names = names
