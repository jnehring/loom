"""Retrying provider calls that failed for transient reasons (network, rate limits, server errors)."""

from __future__ import annotations

import logging
import ssl
import time
from typing import Callable, Optional, TypeVar

T = TypeVar("T")

log = logging.getLogger("loom")

RETRY_STATUS_CODES = frozenset({408, 429, 500, 502, 503, 504})
# Exception class names from the provider SDKs and their HTTP clients (httpx, requests, aiohttp) that mean the
# request never got a usable answer. Matched by name so no SDK has to be importable.
_TRANSIENT_NAMES = frozenset({
    "APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError", "OverloadedError",
    "ConnectError", "ConnectTimeout", "ReadError", "ReadTimeout", "WriteError", "WriteTimeout", "PoolTimeout",
    "RemoteProtocolError", "ConnectionError", "Timeout", "ChunkedEncodingError", "ServerDisconnectedError",
})


def status_code(exc: BaseException) -> Optional[int]:
    """HTTP status of a provider SDK error (OpenAI/Anthropic ``status_code``, google-genai ``code``), else None."""
    for attr in ("status_code", "code"):
        value = getattr(exc, attr, None)
        if isinstance(value, int):
            return value
    return None


def is_rate_limited(exc: BaseException) -> bool:
    """The provider rejected the request outright (HTTP 429), so it was not processed and is safe to resend."""
    return status_code(exc) == 429 or type(exc).__name__ == "RateLimitError"


def is_transient(exc: BaseException) -> bool:
    """Worth retrying: network failure, timeout, rate limit or server error. Client errors (bad model, auth,
    invalid request) are not, they fail the same way again."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        code = status_code(exc)
        if code is not None:
            return code in RETRY_STATUS_CODES
        if isinstance(exc, (ConnectionError, TimeoutError, ssl.SSLError)):
            return True
        if any(cls.__name__ in _TRANSIENT_NAMES for cls in type(exc).__mro__):
            return True
        exc = exc.__cause__ or exc.__context__  # SDKs wrap the httpx/ssl error they got
    return False


def with_retries(
    fn: Callable[[], T],
    *,
    what: str,
    attempts: int = 5,
    base_delay: float = 2.0,
    max_delay: float = 60.0,
    retry_on: Callable[[BaseException], bool] = is_transient,
    sleep: Optional[Callable[[float], None]] = None,
) -> T:
    """Call ``fn``; on an error ``retry_on`` accepts, wait (exponential backoff) and call again, up to ``attempts``
    calls in total. The last error is raised unchanged."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts or not retry_on(exc):
                raise
            delay = min(max_delay, base_delay * 2 ** (attempt - 1))
            log.warning("%s failed (%s: %s), retry %d/%d in %.0f s",
                        what, type(exc).__name__, exc, attempt, attempts - 1, delay)
            (sleep or time.sleep)(delay)
    raise AssertionError("unreachable")
