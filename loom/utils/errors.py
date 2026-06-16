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
