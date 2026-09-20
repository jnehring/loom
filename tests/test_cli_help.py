"""Smoke tests for the CLI help wiring (-h, --help, -?)."""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from loom.main import app

runner = CliRunner()

# Rich help inserts ANSI codes between flag segments (e.g. "--cache-dir"),
# so substring checks must run on plain text.
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def _plain(text: str) -> str:
    return _ANSI_RE.sub("", text)


@pytest.mark.parametrize("flag", ["-h", "--help", "-?"])
def test_top_level_help(flag: str) -> None:
    result = runner.invoke(app, [flag])
    assert result.exit_code == 0
    assert "Loom" in _plain(result.stdout)


@pytest.mark.parametrize("flag", ["-h", "--help", "-?"])
def test_subcommand_help(flag: str) -> None:
    for sub in ("run", "fetch", "list"):
        result = runner.invoke(app, [sub, flag])
        assert result.exit_code == 0, f"{sub} {flag} failed: {result.stdout}"


def test_no_args_prints_help() -> None:
    result = runner.invoke(app, [])
    # no_args_is_help=True returns exit code 2 in Click/Typer
    stdout = _plain(result.stdout)
    stderr = _plain(result.stderr or "")
    assert "Usage" in stdout or "Usage" in stderr


def test_run_help_mentions_cache_dir() -> None:
    result = runner.invoke(app, ["run", "-h"])
    assert result.exit_code == 0
    stdout = _plain(result.stdout)
    assert "--cache-dir" in stdout
    assert "--no-cache" in stdout


def test_cache_clear_help_mentions_cache_dir() -> None:
    result = runner.invoke(app, ["cache", "clear", "-h"])
    assert result.exit_code == 0
    assert "--cache-dir" in _plain(result.stdout)
