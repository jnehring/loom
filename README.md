# Loom: LLM Batch Processing Made Easy

> Weave LLM jobs across OpenAI, Anthropic, Google, and OpenRouter — in batch or live.

## 1. Introduction

Loom is a small Python CLI for running a dataset of prompts (JSON or CSV) through an LLM and merging the responses back into the original file. It speaks two modes:

- **Batch** (`loom run`, default): submits the dataset to the provider's batch API, persists the batch id locally, and later you call `loom fetch` to download and merge results. Cheap (50% off on OpenAI / Anthropic) but asynchronous — can take up to 24 hours.
- **Sequential** (`loom run --sync`): calls the chat-completion endpoint per prompt with a concurrent worker pool, writes the output file immediately, and uses an on-disk response cache.

It also ships a `loom tokens` command that uses each provider's token-counting API where available.

### Supported providers

| Provider        | Batch (`loom run`) | Sequential (`loom run --sync`) | Token counter (`loom tokens`) |
| --------------- | ------------------ | ------------------------------ | ----------------------------- |
| OpenAI          | ✓                  | ✓                              | ✗ — no remote API             |
| Anthropic       | ✓                  | ✓                              | ✓                             |
| Google (Gemini) | ✓                  | ✓                              | ✓                             |
| OpenRouter      | ✗                  | ✓                              | ✗ — no remote API             |

### Table of contents

- [Loom: LLM Batch Processing Made Easy](#loom-llm-batch-processing-made-easy)
  - [1. Introduction](#1-introduction)
    - [Supported providers](#supported-providers)
    - [Table of contents](#table-of-contents)
  - [2. Getting started](#2-getting-started)
    - [Installation](#installation)
    - [Preparing the data](#preparing-the-data)
    - [Submit a batch request](#submit-a-batch-request)
  - [3. Usage](#3-usage)
    - [Command-line reference](#command-line-reference)
      - [`loom run`](#loom-run)
      - [`loom fetch`](#loom-fetch)
      - [`loom list`](#loom-list)
      - [`loom tokens`](#loom-tokens)
      - [`loom cache clear`](#loom-cache-clear)
    - [Batch vs sequential](#batch-vs-sequential)
    - [Storing API keys](#storing-api-keys)
    - [Caching](#caching)
    - [Token counter](#token-counter)
    - [Where Loom stores state](#where-loom-stores-state)
    - [Troubleshooting](#troubleshooting)
  - [4. Developer instructions](#4-developer-instructions)
    - [Repository layout](#repository-layout)
    - [Running unit tests](#running-unit-tests)
    - [Running provider evaluations](#running-provider-evaluations)
    - [GitHub Actions](#github-actions)
    - [Releasing](#releasing)
  - [5. License](#5-license)

## 2. Getting started

### Installation

```bash
pip install loom-batch
```

The PyPI package is `loom-batch` (the name `loom` was taken); the CLI command is `loom`.

From source, for hacking or running tests:

```bash
git clone https://github.com/jannehring/loom
cd loom
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

> **Tip:** if you create the venv with `uv venv`, `pip` is not installed inside it. Use `uv pip install -e ".[dev]"` instead, or recreate the venv with stdlib `python -m venv` (see [Troubleshooting](#troubleshooting)).

### Preparing the data

Loom accepts two input formats — plain or **gzip-compressed** (`.json.gz`, `.csv.gz`). Compressed inputs are decompressed transparently; outputs are always written uncompressed (`.json` / `.csv`).

**JSON** — a list of `{id, prompt}` objects. The `id` is reused as the row key in the merged output.

```json
[
  {"id": "task-001", "prompt": "Summarize the plot of Hamlet in one sentence."},
  {"id": "task-002", "prompt": "Translate 'Good morning' to French."}
]
```

**CSV** — any schema; you tell Loom which column holds the prompt with `--col`. All original columns are preserved; a new `llm_response` column is appended.

```csv
id,text,priority
1,"Explain quantum physics in one paragraph",low
2,"Write a haiku about rust",high
```

### Submit a batch request

Minimal end-to-end run, passing the API key inline (see [Storing API keys](#storing-api-keys) for cleaner options):

```bash
loom run --file prompts.json \
         --provider openai \
         --model gpt-4o-mini \
         --api-key sk-...
# -> Batch submitted. id=batch_abc123 provider=openai

# ...minutes or hours later...
loom fetch              # --all is the default; fetches every pending batch
# -> Fabric complete. id=batch_abc123 -> prompts_results_openai_gpt-4o-mini.json
```

The output is written next to the input as `<name>_results_<provider>_<model>.<ext>`. Forward slashes and other unsafe characters in the model id are replaced with underscores (e.g. `openai/gpt-4o-mini` → `openai_gpt-4o-mini`). For gzipped inputs the `.gz` is dropped — `data.csv.gz` → `data_results_<provider>_<model>.csv`. Override the path entirely with `--output`.

## 3. Usage

### Command-line reference

#### `loom run`

Submit a dataset as a batch job (default) or run it synchronously with `--sync`.

| Flag                 | Default                                    | Description                                                                                                            |
| -------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `--file`, `-f`       | _required_                                 | Input `.json`, `.csv`, `.json.gz`, or `.csv.gz`.                                                                       |
| `--provider`, `-p`   | _required_                                 | `openai`, `anthropic`, `google`, or `openrouter`.                                                                      |
| `--model`, `-m`      | _required_                                 | Provider-specific model id (e.g. `gpt-4o-mini`, `claude-3-5-sonnet-latest`, `gemini-2.0-flash`, `openai/gpt-4o-mini`). |
| `--col`, `-c`        | —                                          | Prompt column name (required for CSV).                                                                                 |
| `--api-key`          | env / `.env`                               | Override the resolved API key for this run.                                                                            |
| `--output`, `-o`     | `<input>_results_<provider>_<model>.<ext>` | Custom output file path.                                                                                               |
| `--sync` / `--batch` | `--batch`                                  | `--sync` calls the provider per prompt and writes the output immediately. `--batch` uses the provider's batch API.     |
| `--workers`, `-w`    | `8`                                        | Concurrent workers in `--sync` mode.                                                                                   |
| `--no-cache`         | off                                        | Disable the on-disk response cache (`--sync` only).                                                                    |
| `--force`            | off                                        | Overwrite an existing output file without prompting (`--sync` only).                                                   |
| `--with-meta`        | off                                        | Add `llm_provider` and `llm_model` columns (CSV) or fields (JSON) to the output, alongside `llm_response`.             |

OpenRouter has no batch API; using `--provider openrouter` without `--sync` exits with a helpful error.

#### `loom fetch`

Poll the provider, download results, merge into the output file.

| Flag                       | Default      | Description                                                                                                              |
| -------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `--id`, `-i`               | —            | Fetch a single batch by id. If set, implies `--no-all`.                                                                  |
| `--all` / `--no-all`, `-a` | `--all`      | Process every pending batch under `~/.loom/batches/`. This is the default — `loom fetch` with no args walks all batches. |
| `--api-key`                | env / `.env` | Override the resolved API key.                                                                                           |
| `--keep`, `-k`             | off          | Keep the metadata file in `~/.loom/batches/` after a successful fetch (default: delete it).                              |
| `--force`                  | off          | Overwrite existing output files without prompting.                                                                       |

For pending batches, `loom fetch` prints the current status and a one-sentence explanation. The full set of possible statuses:

| Status        | Meaning                                                                                                                                                                                                                         |
| ------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `validating`  | Provider has accepted the batch and is queueing/preparing it; no work has started yet.                                                                                                                                          |
| `in_progress` | Provider is actively running the prompts; check back later.                                                                                                                                                                     |
| `completed`   | All prompts finished and results were downloaded — the merged output file has been written.                                                                                                                                     |
| `failed`      | Provider reported the batch as failed; results are not available.                                                                                                                                                               |
| `expired`     | Batch exceeded the provider's time limit (typically 24h) before completing.                                                                                                                                                     |
| `cancelled`   | Batch was cancelled — either by you on the provider's dashboard, or by the provider itself.                                                                                                                                     |
| `unknown`     | The last fetch attempt raised an error (invalid id, auth failure, network glitch, or an API response Loom doesn't recognise). Re-run `loom fetch` to retry; if it persists, inspect the metadata file under `~/.loom/batches/`. |

`validating` and `in_progress` are the only non-terminal states — `loom fetch` will pick the batch up again on the next run. The other states are terminal: `completed` means the output file is on disk, and `failed` / `expired` / `cancelled` mean no merge happened.

#### `loom list`

List every batch known to Loom, with last-seen status, model, and source file. No flags.

#### `loom tokens`

Count input tokens for every prompt using the provider's token-counting API. See [Token counter](#token-counter).

| Flag               | Default      | Description                                       |
| ------------------ | ------------ | ------------------------------------------------- |
| `--file`, `-f`     | _required_   | Input `.json`, `.csv`, `.json.gz`, or `.csv.gz`.  |
| `--provider`, `-p` | _required_   | `openai`, `anthropic`, `google`, or `openrouter`. |
| `--model`, `-m`    | _required_   | Provider-specific model id.                       |
| `--col`, `-c`      | —            | Prompt column name (required for CSV).            |
| `--api-key`        | env / `.env` | Override the resolved API key.                    |
| `--workers`, `-w`  | `8`          | Concurrent workers.                               |

#### `loom cache clear`

Delete every cached response under `~/.loom/cache/`. See [Caching](#caching).

| Flag          | Default | Description                   |
| ------------- | ------- | ----------------------------- |
| `--yes`, `-y` | off     | Skip the confirmation prompt. |

### Batch vs sequential

|                             | `loom run` (batch, default) | `loom run --sync` (sequential) |
| --------------------------- | --------------------------- | ------------------------------ |
| Latency                     | Up to 24h                   | Real-time                      |
| Pricing (OpenAI, Anthropic) | 50% off                     | Standard                       |
| Steps                       | `run` → wait → `fetch`      | Single command                 |
| Cache                       | n/a                         | On-disk, on by default         |
| OpenRouter                  | ✗                           | ✓ (only mode)                  |
| State on disk               | `~/.loom/batches/`          | None (cache only)              |

Pick **batch** when you have a large dataset and don't care about wall-clock time. Pick **sync** when you want results now, or when the provider has no batch API (OpenRouter).

### Storing API keys

Loom resolves keys in this order: **`--api-key` flag → environment variable → `.env` file** in the current working directory (loaded via `python-dotenv`, does not overwrite existing env vars).

Recognised environment variables:

```ini
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
OPENROUTER_API_KEY=sk-or-...
```

A `.env` in the working directory is the friction-free option for daily use; `--api-key` is handy for one-offs or shared workstations.

### Caching

In `--sync` mode, Loom caches every response under `~/.loom/cache/`. The cache key is `sha256("<provider>|<model>|<prompt>")`, so changing any of those misses the cache. There is no TTL or eviction — the cache grows monotonically until you clear it.

```bash
loom run --sync -p openai -m gpt-4o-mini -f data.csv -c text   # first run: API calls
loom run --sync -p openai -m gpt-4o-mini -f data.csv -c text   # second run: 100% cache hits
loom run --sync -p openai -m gpt-4o-mini -f data.csv -c text --no-cache  # bypass
loom cache clear                                                # wipe ~/.loom/cache/
```

`loom run --sync` reports cache hits live in its progress bar.

### Token counter

```bash
loom tokens --file prompts.json --provider anthropic --model claude-3-5-sonnet-latest
# Counting tokens ████████░░░░  340/1000  est_total≈36,210  errors=0  0:01:12  eta 0:02:35
# -> Total input tokens: 12,345 across 100 prompts (provider=anthropic, model=claude-3-5-sonnet-latest, errors=0)
```

`loom tokens` calls each provider's official count-tokens endpoint, one prompt at a time, with a concurrent worker pool. The live progress bar shows:

- `done/total` prompts processed,
- `est_total` — running estimate of the final input-token count, computed as the mean tokens-per-prompt-so-far multiplied by `total` (refines as more prompts complete),
- `errors`,
- elapsed time and `eta` (estimated time remaining, based on the current rate).

| Provider   | Endpoint                                             | Available                                 |
| ---------- | ---------------------------------------------------- | ----------------------------------------- |
| Anthropic  | `client.messages.count_tokens(...)` → `input_tokens` | ✓                                         |
| Google     | `client.models.count_tokens(...)` → `total_tokens`   | ✓                                         |
| OpenAI     | —                                                    | ✗ (no remote API; use `tiktoken` locally) |
| OpenRouter | —                                                    | ✗                                         |

For unsupported providers, `loom tokens` prints _"Token counting not available: ..."_ and exits with code 2.

### Where Loom stores state

```
~/.loom/
├── batches/        # one <provider>_<batch_id>.json per pending or kept batch
└── cache/          # one <sha256>.json per cached --sync response
```

- `~/.loom/batches/<provider>_<safe_id>.json` is created by `loom run` (batch mode) and contains `batch_id`, `provider`, `model`, `original_file_path`, `file_type`, the prompt column, an `id_map` mapping internal `custom_id` → original row id, `created_at`, and the last-seen `status`. `loom fetch` updates `status`, downloads results, and (unless `--keep` is passed) deletes the file on success.
- `~/.loom/cache/<sha256>.json` is the response cache used by `--sync`. Each file holds `{provider, model, response, created_at}`.

Both directories are safe to delete by hand: cache will rebuild itself; deleting `batches/` orphans any in-flight batch jobs (they still complete on the provider's side, you just lose Loom's view of them).

### Troubleshooting

**`ModuleNotFoundError: No module named 'pandas'` right after `pip install -e ".[dev]"`**
Your `.venv` was probably created with `uv venv`, which doesn't install `pip` inside. Your `pip install` ran against the system / conda `pip` and dropped the packages elsewhere. Fix with `uv pip install -e ".[dev]"`, or recreate the venv with `python -m venv .venv && source .venv/bin/activate && pip install -e ".[dev]"`.

**`which loom` shows `/opt/miniconda3/bin/loom` even after `source .venv/bin/activate`**
Conda's path is being prepended after the venv. Either reorder your shell init, or just call the venv binary directly: `./.venv/bin/loom <cmd>`.

**Google batch results mapped to the wrong rows**
Google's batch API can return inlined responses out of submission order (especially for 100+ requests). Loom matches each response using `metadata.custom_id` from the request — not list position. If you see mismatched results from an older Loom version, upgrade (`pip install -U loom-batch`) and re-submit the batch. Requires `google-genai>=1.61.0`, which restores metadata on batch responses.

**Google batch `Invalid batch job name: jqpem7...`**
The stored `batch_id` is missing the required `batches/` prefix (an older Loom version stripped it). Edit `~/.loom/batches/google_<id>.json`, change `"batch_id": "<id>"` to `"batch_id": "batches/<id>"`, and rename the file to `google_batches_<id>.json` so the on-disk filename and the in-file id stay consistent.

**`status=unknown` in `loom fetch`**
The previous fetch attempt raised an exception (bad id, network blip, expired key, or an API response Loom doesn't recognise). Re-running `loom fetch` retries; if it persists, run with the provider's SDK directly to surface the underlying error.

**`Error: OpenRouter has no batch API`**
OpenRouter doesn't offer batch processing. Re-run with `--sync`.

## 4. Developer instructions

### Repository layout

```
loom/
  main.py                       # CLI entry point (Typer commands)
  core/
    orchestrator.py             # run_batch, fetch_batch, generate_sync, count_tokens
    models.py                   # Pydantic models, ProviderName, BatchStatus
  eval/
    eval_providers.py           # Provider evaluation script (init / fetch)
  providers/
    base.py                     # Batch provider ABC (submit/check_status/download)
    sync_base.py                # Sync provider ABC (generate/count_tokens)
    openai.py, anthropic.py,
    google.py                   # Batch implementations
    openai_sync.py, anthropic_sync.py,
    google_sync.py, openrouter_sync.py   # Sync implementations
  utils/
    converters.py               # Load / merge JSON & CSV
    storage.py                  # ~/.loom/batches/ persistence
    cache.py                    # ~/.loom/cache/ response cache
    keys.py                     # API-key resolution
tests/                          # pytest suite
.github/workflows/              # CI: test.yml, publish.yml
pyproject.toml                  # Dependencies and package metadata
```

### Running unit tests

```bash
pip install -e ".[dev]"
pytest                 # quiet
pytest -v              # verbose
pytest tests/test_converters.py     # one file
pytest tests/test_storage.py::test_save_and_load_roundtrip   # one test
```

### Running provider evaluations

Loom includes a provider-level evaluation script to test both synchronous and batch APIs for all supported providers using a small dataset of 3 prompts with predictable single-word outputs. This requires the API keys to be configured and it generates costs.

```bash
# 1. Initialize evaluation: test sync APIs and submit batch jobs (default: all providers)
python -m loom.eval.eval_providers init

# Alternatively, initialize for a single provider (e.g. google, openai, or anthropic)
python -m loom.eval.eval_providers init google

# 2. Fetch evaluation results: check batch statuses and download/validate results (default: all providers)
python -m loom.eval.eval_providers fetch

# Alternatively, fetch for a single provider only
python -m loom.eval.eval_providers fetch google
```

This runs against live provider APIs. Configure your API keys (e.g. `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `ANTHROPIC_API_KEY`) in your environment or a `.env` file before running. Any provider without a configured API key will be skipped automatically.


### GitHub Actions

- [`.github/workflows/test.yml`](.github/workflows/test.yml) — runs on every push and PR, with a matrix over Python 3.10 / 3.11 / 3.12. Installs the project with `pip install -e ".[dev]"` and runs `pytest -v`.
- [`.github/workflows/publish.yml`](.github/workflows/publish.yml) — manual release workflow (`workflow_dispatch`). Pick **patch**, **minor**, or **major**, and it bumps `pyproject.toml` + `loom/__init__.py`, runs tests, builds an sdist + wheel, commits and tags the release, creates a GitHub Release, and uploads to PyPI via **OIDC Trusted Publishing** — no PyPI token is stored in repo secrets.

### Releasing

**One-time PyPI setup:**

1. Create the project on PyPI: https://pypi.org/manage/account/publishing/
2. Add a **trusted publisher** with:
   - Owner: `jannehring`
   - Repository: `loom`
   - Workflow: `publish.yml`
   - Environment: `pypi`
3. In GitHub: **Settings → Environments → New environment → `pypi`**. Enable manual approval here if you want a human in the loop for every release.

**Cutting a release:**

1. Open **Actions → publish → Run workflow** on `main`.
2. Choose **patch**, **minor**, or **major** (e.g. `0.1.0` → `0.1.1` / `0.2.0` / `1.0.0`).
3. The workflow bumps the version, runs tests, builds, commits `Release vX.Y.Z`, pushes the tag, creates a GitHub Release, and publishes to PyPI.

The version bump is only pushed if tests and the build succeed.

## 5. License

MIT — see [LICENSE](LICENSE).
