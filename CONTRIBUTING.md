# Contributing to Loom

### Table of contents

[Setting up](#setting-up)\
[Repository layout](#repository-layout)\
[Running unit tests](#running-unit-tests)\
[Running provider evaluations](#running-provider-evaluations)\
[GitHub Actions](#github-actions)\
[Releasing](#releasing)

## Setting up

```bash
git clone https://github.com/jnehring/loom
cd loom
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

> **Tip:** a venv created with `uv venv` has no `pip` inside it. Use `uv pip install -e ".[dev]"` instead, or create the venv with `python -m venv` as above.

## Repository layout

```
loom/
  __init__.py                   # Public exports (Loom, result types, …)
  api.py                        # Loom client (library API)
  main.py                       # CLI entry point (Typer commands)
  core/
    orchestrator.py             # run_batch, fetch_batch, generate_sync, count_tokens, generate_items
    models.py                   # Pydantic models, ProviderName, BatchStatus, GenerationParams
  eval/
    eval_providers.py           # Provider evaluation script (init / fetch)
  providers/
    base.py                     # Batch provider ABC (submit/check_status/download)
    sync_base.py                # Sync provider ABC (generate/count_tokens)
    openai.py, anthropic.py,
    google.py                   # Batch implementations
    openai_sync.py, anthropic_sync.py,
    google_sync.py, openrouter_sync.py   # Sync implementations
    alibaba.py                  # Alibaba Cloud batch + sync (OpenAI-compatible API)
    params.py                   # GenerationParams → provider request fields
  utils/
    converters.py               # Load / merge JSON, CSV & Parquet
    storage.py                  # ~/.loom/batches/ persistence
    cache.py                    # ResponseCache (configurable directory)
    paths.py                    # LOOM_HOME / LOOM_CACHE_DIR resolution
    keys.py                     # API-key resolution
    errors.py                   # Exceptions (e.g. UnsupportedParameterError)
docs/
  api.md                        # Python library API reference
  batch-jobs.md                 # Batch statuses and on-disk state
tests/                          # pytest suite
.github/workflows/              # CI: test.yml, publish.yml
pyproject.toml                  # Dependencies and package metadata
```

## Running unit tests

```bash
pytest                 # quiet
pytest -v              # verbose
pytest tests/test_converters.py     # one file
pytest tests/test_storage.py::test_save_and_load_roundtrip   # one test
```

## Running provider evaluations

`loom/eval/eval_providers.py` tests the sync and batch APIs of OpenAI, Anthropic, and Google against live provider APIs, using a dataset of 3 prompts with predictable single-word outputs. It needs API keys and **costs money**.

```bash
# 1. Test the sync APIs and submit batch jobs (default: all providers)
python -m loom.eval.eval_providers init

# ...or for a single provider (google, openai, or anthropic)
python -m loom.eval.eval_providers init google

# 2. Check batch statuses, download and validate results (default: all providers)
python -m loom.eval.eval_providers fetch

# ...or for a single provider only
python -m loom.eval.eval_providers fetch google
```

Configure `OPENAI_API_KEY`, `GOOGLE_API_KEY`, and `ANTHROPIC_API_KEY` in your environment or a `.env` file first. Providers without a key are skipped.

## GitHub Actions

- [`.github/workflows/test.yml`](.github/workflows/test.yml) — runs on every push and PR, with a matrix over Python 3.10 / 3.11 / 3.12. Installs the project with `pip install -e ".[dev]"` and runs `pytest -v`.
- [`.github/workflows/publish.yml`](.github/workflows/publish.yml) — manual release workflow (`workflow_dispatch`). Pick **patch**, **minor**, or **major**, and it bumps `pyproject.toml` + `loom/__init__.py`, runs tests, builds an sdist + wheel, commits and tags the release, creates a GitHub Release, and uploads to PyPI via **OIDC Trusted Publishing** — no PyPI token is stored in repo secrets.

## Releasing

1. Open **Actions → publish → Run workflow** on `main`.
2. Choose **patch**, **minor**, or **major** (e.g. `0.1.0` → `0.1.1` / `0.2.0` / `1.0.0`).
3. The workflow bumps the version, runs tests, builds, commits `Release vX.Y.Z`, pushes the tag, creates a GitHub Release, and publishes to PyPI.

The version bump is only pushed if tests and the build succeed.
