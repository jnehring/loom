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


def test_dotenv_found_in_working_directory(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A .env in the caller's working directory is loaded, even when loom is installed elsewhere."""
    import subprocess
    import sys

    (tmp_path / ".env").write_text("GOOGLE_API_KEY=from-dotenv\n")
    script = tmp_path / "script.py"  # run as a file: that is when python-dotenv searches from the caller's location
    script.write_text("from loom.utils import keys\nprint(keys.resolve_api_key('google'))\n")
    out = subprocess.run([sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "from-dotenv"
