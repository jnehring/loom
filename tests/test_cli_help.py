"""Smoke tests for the CLI help wiring (-h, --help, -?)."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from loom.main import app

runner = CliRunner()


@pytest.mark.parametrize("flag", ["-h", "--help", "-?"])
def test_top_level_help(flag: str) -> None:
    result = runner.invoke(app, [flag])
    assert result.exit_code == 0
    assert "Loom" in result.stdout


@pytest.mark.parametrize("flag", ["-h", "--help", "-?"])
def test_subcommand_help(flag: str) -> None:
    for sub in ("run", "fetch", "list"):
        result = runner.invoke(app, [sub, flag])
        assert result.exit_code == 0, f"{sub} {flag} failed: {result.stdout}"


def test_no_args_prints_help() -> None:
    result = runner.invoke(app, [])
    # no_args_is_help=True returns exit code 2 in Click/Typer
    assert "Usage" in result.stdout or "Usage" in (result.stderr or "")
