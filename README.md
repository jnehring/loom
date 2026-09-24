# Loom: LLM Batch Processing Made Easy

<img src="https://github.com/jnehring/loom/blob/main/logos/loom-logo-small.png" width="250" style="float:left">

Weave LLM jobs across OpenAI, Anthropic, Google, and OpenRouter — in batch or live.

## 1. Introduction

Loom is a small Python CLI **and library** for running a dataset of prompts (JSON, CSV, or Parquet) through an LLM and merging the responses back into the original file. It speaks two modes:

- **Batch** (`loom run`, default): submits the dataset to the provider's batch API, persists the batch id locally, and later you call `loom fetch` to download and merge results. Cheap (50% off on OpenAI / Anthropic) but asynchronous — can take up to 24 hours. Uses the same on-disk response cache as sync (skips cached prompts at submit, writes downloads into the cache on fetch).
- **Sequential** (`loom run --sync`): calls the chat-completion endpoint per prompt with a concurrent worker pool, writes the output file immediately, and uses an on-disk response cache.

It also ships a `loom tokens` command that uses each provider's token-counting API where available, and a [`Loom`](docs/api.md) Python client for in-memory and file-based use without the CLI.

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
    - [Python library](#python-library)
  - [3. Usage](#3-usage)
    - [Command-line reference](#command-line-reference)
      - [`loom run`](#loom-run)
      - [`loom fetch`](#loom-fetch)
      - [`loom list`](#loom-list)
      - [`loom tokens`](#loom-tokens)
      - [`loom cache clear`](#loom-cache-clear)
    - [Batch vs sequential](#batch-vs-sequential)
    - [Generation settings](#generation-settings)
    - [Storing API keys](#storing-api-keys)
    - [Caching](#caching)
    - [Token counter](#token-counter)
    - [Where Loom stores state](#where-loom-stores-state)
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

Loom accepts three input formats — plain or **gzip-compressed** (`.json.gz`, `.csv.gz`). Compressed inputs are decompressed transparently; JSON and CSV outputs are always written uncompressed (`.json` / `.csv`). Parquet inputs produce `.parquet` output.

**JSON** — a list of `{id, prompt}` objects. The `id` is reused as the row key in the merged output.

```json
[
  {"id": "task-001", "prompt": "Summarize the plot of Hamlet in one sentence."},
  {"id": "task-002", "prompt": "Translate 'Good morning' to French."}
]
```

**CSV** — any schema; Loom reads the prompt from the `text` column by default (override with `--col`). All original columns are preserved; a new `llm_response` column is appended.

```csv
id,text,priority
1,"Explain quantum physics in one paragraph",low
2,"Write a haiku about rust",high
```

**Parquet** — same semantics as CSV: reads the `text` column by default (override with `--col`). All original columns are preserved; `llm_response` is appended. Output is written as `.parquet`.

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

### Python library

```python
from loom import Loom

client = Loom("openai", "gpt-4o-mini", cache_dir="/tmp/loom-cache", temperature=0.2, max_tokens=500)

# In-memory
print(client.generate("Say hello"))
print(client.generate("Write a haiku", params={"temperature": 1.0}))  # per-call override
result = client.generate_many(["a", "b", "c"])
print(result.texts, result.cache_hits)

# File-based sync
run = client.run_file("prompts.json", force=True)

# Batch
job = client.submit_file("prompts.csv", column="text")
fetch = job.fetch()  # later
```

Full reference: **[docs/api.md](docs/api.md)**.

## 3. Usage

### Command-line reference

#### `loom run`

Submit a dataset as a batch job (default) or run it synchronously with `--sync`.

| Flag                 | Default                                    | Description                                                                                                            |
| -------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- |
| `--file`, `-f`       | _required_                                 | Input `.json`, `.csv`, `.parquet`, `.json.gz`, or `.csv.gz`.                                                           |
| `--provider`, `-p`   | _required_                                 | `openai`, `anthropic`, `google`, or `openrouter`.                                                                      |
| `--model`, `-m`      | _required_                                 | Provider-specific model id (e.g. `gpt-4o-mini`, `claude-3-5-sonnet-latest`, `gemini-2.0-flash`, `openai/gpt-4o-mini`). |
| `--col`, `-c`        | `text`                                     | Prompt column name (CSV and Parquet).                                                                                  |
| `--api-key`          | env / `.env`                               | Override the resolved API key for this run.                                                                            |
| `--output`, `-o`     | `<input>_results_<provider>_<model>.<ext>` | Custom output file path.                                                                                               |
| `--sync` / `--batch` | `--batch`                                  | `--sync` calls the provider per prompt and writes the output immediately. `--batch` uses the provider's batch API.     |
| `--workers`, `-w`    | `8`                                        | Concurrent workers in `--sync` mode.                                                                                   |
| `--no-cache`         | off                                        | Disable the on-disk response cache (both `--sync` and `--batch`).                                                          |
| `--cache-dir`        | `$LOOM_CACHE_DIR` / `~/.loom/cache`        | Override the response-cache directory.                                                                                     |
| `--force`            | off                                        | Overwrite an existing output file without prompting.                                                                       |
| `--with-meta`        | off                                        | Add `llm_provider` and `llm_model` columns (CSV/Parquet) or fields (JSON) to the output, alongside `llm_response`.     |
| `--temperature`, `-t`, `--max-tokens`, `--top-p`, `--top-k`, `--stop`, `--seed`, `--presence-penalty`, `--frequency-penalty`, `--system`, `--json`, `--param` | provider default | Generation settings, see [Generation settings](#generation-settings). |

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
| `--file`, `-f`     | _required_   | Input `.json`, `.csv`, `.parquet`, `.json.gz`, or `.csv.gz`. |
| `--provider`, `-p` | _required_   | `openai`, `anthropic`, `google`, or `openrouter`. |
| `--model`, `-m`    | _required_   | Provider-specific model id.                       |
| `--col`, `-c`      | `text`       | Prompt column name (CSV and Parquet). |
| `--api-key`        | env / `.env` | Override the resolved API key.                    |
| `--workers`, `-w`  | `8`          | Concurrent workers.                               |

#### `loom cache clear`

Delete every cached response under the cache directory (default `~/.loom/cache/`). See [Caching](#caching).

| Flag          | Default | Description                   |
| ------------- | ------- | ----------------------------- |
| `--yes`, `-y` | off     | Skip the confirmation prompt. |
| `--cache-dir` | default | Override the cache directory. |

### Batch vs sequential

|                             | `loom run` (batch, default) | `loom run --sync` (sequential) |
| --------------------------- | --------------------------- | ------------------------------ |
| Latency                     | Up to 24h                   | Real-time                      |
| Pricing (OpenAI, Anthropic) | 50% off                     | Standard                       |
| Steps                       | `run` → wait → `fetch`      | Single command                 |
| Cache                       | On-disk (skip at submit, write on fetch) | On-disk, on by default |
| OpenRouter                  | ✗                           | ✓ (only mode)                  |
| State on disk               | `~/.loom/batches/`          | None (cache only)              |

Pick **batch** when you have a large dataset and don't care about wall-clock time. Pick **sync** when you want results now, or when the provider has no batch API (OpenRouter).

### Generation settings

Temperature, output length and the other common sampling options work the same way for every provider. Loom
translates them into each provider's request format:

| Setting             | CLI flag               | OpenAI                  | Anthropic        | Google (Gemini)       | OpenRouter         |
| ------------------- | ---------------------- | ----------------------- | ---------------- | --------------------- | ------------------ |
| `temperature`       | `--temperature`, `-t`  | `temperature`           | `temperature`    | `temperature`         | `temperature`      |
| `max_tokens`        | `--max-tokens`         | `max_completion_tokens` | `max_tokens` ¹   | `max_output_tokens`   | `max_tokens`       |
| `top_p`             | `--top-p`              | `top_p`                 | `top_p`          | `top_p`               | `top_p`            |
| `top_k`             | `--top-k`              | ✗                       | `top_k`          | `top_k`               | `top_k`            |
| `stop`              | `--stop` (repeatable)  | `stop`                  | `stop_sequences` | `stop_sequences`      | `stop`             |
| `seed`              | `--seed`               | `seed`                  | ✗                | `seed`                | `seed`             |
| `presence_penalty`  | `--presence-penalty`   | `presence_penalty`      | ✗                | `presence_penalty`    | `presence_penalty` |
| `frequency_penalty` | `--frequency-penalty`  | `frequency_penalty`     | ✗                | `frequency_penalty`   | `frequency_penalty`|
| `system`            | `--system`             | system message          | `system`         | `system_instruction`  | system message     |
| `json_mode`         | `--json`               | `response_format` JSON  | ✗                | `response_mime_type`  | `response_format`  |
| `extra`             | `--param key=value`    | request body            | message params   | `GenerateContentConfig` | request body     |

¹ Anthropic requires `max_tokens`; Loom sends 4096 when it is not set.

A setting marked ✗ raises `UnsupportedParameterError` before anything is sent — Loom never drops a setting silently.
Unset settings are not sent, so the provider default applies. Batch and sync requests carry exactly the same settings.
`extra` passes provider-specific options through unchanged (e.g. `reasoning_effort` for OpenAI reasoning models,
`thinking_config` for Gemini); Loom does not validate them.

```bash
loom run -p google -m gemini-2.0-flash -f data.csv -t 0.2 --max-tokens 800 --system "Answer in German." --json
loom run -p anthropic -m claude-3-5-sonnet-latest -f data.csv --temperature 0 --stop "###"
loom run -p openai -m o4-mini -f data.csv --param reasoning_effort=low
```

```python
from loom import Loom, GenerationParams

client = Loom("google", "gemini-2.0-flash", temperature=0.2, max_tokens=800)
client.generate("…", params={"temperature": 0.9})        # override for one call
client = Loom("openai", "gpt-4o-mini", params=GenerationParams(seed=7, json_mode=True))
```

Settings are part of the cache key (see [Caching](#caching)): the same prompt at another temperature is a new request.

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

Loom caches every response under `~/.loom/cache/` (override with `--cache-dir`, `$LOOM_CACHE_DIR`, or `$LOOM_HOME`). The cache key is `sha256("<provider>|<model>|<prompt>|<settings>")`, so changing any of those misses the cache. `<settings>` is the canonical JSON of the [generation settings](#generation-settings) that are set; with no settings it is left out, so caches from Loom ≤ 0.4 stay valid. There is no TTL or eviction — the cache grows monotonically until you clear it.

**Sync mode** reads the cache before calling the provider and writes every successful response.

**Batch mode** also uses the cache:

- at submit time, cached prompts are skipped (only misses go to the provider);
- if *every* prompt is cached, Loom writes the output immediately and skips the provider entirely;
- at fetch time, newly downloaded responses are written into the cache.

```bash
loom run --sync -p openai -m gpt-4o-mini -f data.csv -c text   # first run: API calls
loom run --sync -p openai -m gpt-4o-mini -f data.csv -c text   # second run: 100% cache hits
loom run -p openai -m gpt-4o-mini -f data.csv -c text          # batch: skips cached prompts
loom run --sync -p openai -m gpt-4o-mini -f data.csv --no-cache
loom run --sync -p openai -m gpt-4o-mini -f data.csv --cache-dir /tmp/my-cache
loom cache clear                                                # wipe the cache directory
loom cache clear --cache-dir /tmp/my-cache
```

`loom run --sync` reports cache hits live in its progress bar. From Python, pass `cache_dir=` to [`Loom`](docs/api.md#cache-configuration).

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
~/.loom/                    # or $LOOM_HOME
├── batches/                # one <provider>_<batch_id>.json per pending or kept batch
├── cache/                  # one <sha256>.json per cached response ($LOOM_CACHE_DIR overrides)
└── inputs/                 # prompt snapshots for in-memory batch submits (library API)
```

- `~/.loom/batches/<provider>_<safe_id>.json` is created by `loom run` (batch mode) and contains `batch_id`, `provider`, `model`, `original_file_path`, `file_type`, the prompt column, an `id_map` mapping internal `custom_id` → original row id, `created_at`, the last-seen `status`, plus any responses already served from cache at submit time. `loom fetch` updates `status`, downloads results, writes them into the cache, and (unless `--keep` is passed) deletes the file on success.
- `~/.loom/cache/<sha256>.json` is the response cache used by both `--sync` and batch. Each file holds `{provider, model, response, created_at}`.
- `~/.loom/inputs/` holds temporary JSON snapshots for batches submitted via the Python `Loom.submit(...)` API.

Both `batches/` and `cache/` are safe to delete by hand: cache will rebuild itself; deleting `batches/` orphans any in-flight batch jobs (they still complete on the provider's side, you just lose Loom's view of them).

## 4. Developer instructions

### Repository layout

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
    params.py                   # GenerationParams → provider request fields
  utils/
    converters.py               # Load / merge JSON, CSV & Parquet
    storage.py                  # ~/.loom/batches/ persistence
    cache.py                    # ResponseCache (configurable directory)
    paths.py                    # LOOM_HOME / LOOM_CACHE_DIR resolution
    keys.py                     # API-key resolution
docs/
  api.md                        # Python library API reference
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

1. Open **Actions → publish → Run workflow** on `main`.
2. Choose **patch**, **minor**, or **major** (e.g. `0.1.0` → `0.1.1` / `0.2.0` / `1.0.0`).
3. The workflow bumps the version, runs tests, builds, commits `Release vX.Y.Z`, pushes the tag, creates a GitHub Release, and publishes to PyPI.

The version bump is only pushed if tests and the build succeed.

## 5. License

MIT — see [LICENSE](LICENSE).
