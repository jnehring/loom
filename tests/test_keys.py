"""API key resolution precedence."""

from __future__ import annotations

import pytest

from loom.utils import keys


def test_explicit_flag_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-env")
    assert keys.resolve_api_key("openai", explicit="from-flag") == "from-flag"


def test_env_var_used(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    assert keys.resolve_api_key("anthropic") == "env-key"


def test_missing_key_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="No API key"):
        keys.resolve_api_key("google")
