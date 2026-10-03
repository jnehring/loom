# Loom: LLM Batch Processing Made Easy

<p align="center"><img src="https://raw.githubusercontent.com/jnehring/loom/main/logos/loom-logo-small.png" width="250" alt="Loom logo"></p>

Run a whole dataset of prompts through any major LLM with one command, and get your file back with the answers in a new column.

## 1. Introduction

Loom is a Python CLI **and library** for running a dataset of prompts (JSON, CSV, or Parquet) through an LLM. It writes the responses into a copy of your file and keeps every original column.

- **One interface, five providers.** OpenAI, Anthropic, Google Gemini, OpenRouter, and Alibaba Cloud (Qwen). Switching provider means changing `--provider` and `--model`; the input file and the command stay the same.
- **Batch or live.** `loom run` submits the dataset to the provider's batch API: 50% cheaper on OpenAI and Anthropic, with results within 24 hours that you collect with `loom fetch`. `loom run --sync` calls the API live with a pool of concurrent workers and writes the output right away.
- **No paying twice for the same prompt.** Every response is cached on disk. Re-runs, interrupted runs, and batch submissions skip prompts that were already answered.
- **The same generation settings everywhere.** Temperature, max tokens, system prompt, JSON mode, and more are translated into each provider's request format. A setting the provider doesn't support raises an error instead of being dropped silently.
- **Token counts before you spend.** `loom tokens` counts the input tokens of a dataset with the provider's own token-counting API.
- **CLI or Python.** The [`Loom`](docs/api.md) client offers the same features for in-memory prompts and for files.

### Example

Classify the sentiment of a CSV of reviews:

```csv
id,text
1,"The battery died after two days."
2,"Fast shipping, works perfectly."
```

```bash
loom run --sync -f reviews.csv -p openai -m gpt-5.4-mini \
         --system "Classify the sentiment as positive or negative. Answer with one word."
```

Loom writes `reviews_results_openai_gpt-5.4-mini.csv`, with the answers in a new `llm_response` column:

```csv
id,text,llm_response
1,"The battery died after two days.",negative
2,"Fast shipping, works perfectly.",positive
```

Drop `--sync` to send the same file through the provider's batch API at half the price.

### Supported providers

| Provider                           | Batch (`loom run`) | Sequential (`loom run --sync`) | Token counter (`loom tokens`) |
| ---------------------------------- | ------------------ | ------------------------------ | ----------------------------- |
| OpenAI                             | ✓                  | ✓                              | ✓                             |
| Anthropic                          | ✓                  | ✓                              | ✓                             |
| Google (Gemini)                    | ✓                  | ✓                              | ✓                             |
| OpenRouter                         | ✗                  | ✓                              | ✗ — no remote API             |
| Alibaba Cloud (Model Studio, Qwen) | ✓                  | ✓                              | ✗ — no remote API             |

### Table of contents

[1. Introduction](#1-introduction)\
&emsp;[Example](#example)\
&emsp;[Supported providers](#supported-providers)\
[2. Getting started](#2-getting-started)\
&emsp;[Installation](#installation)\
&emsp;[Preparing the data](#preparing-the-data)\
&emsp;[Submit a batch request](#submit-a-batch-request)\
&emsp;[Python library](#python-library)\
[3. Usage](#3-usage)\
&emsp;[Command-line reference](#command-line-reference)\
&emsp;&emsp;[`loom run`](#loom-run)\
&emsp;&emsp;[`loom fetch`](#loom-fetch)\
&emsp;&emsp;[`loom list`](#loom-list)\
&emsp;&emsp;[`loom tokens`](#loom-tokens)\
&emsp;&emsp;[`loom cache clear`](#loom-cache-clear)\
&emsp;[Batch vs sequential](#batch-vs-sequential)\
&emsp;[Generation settings](#generation-settings)\
&emsp;[Storing API keys](#storing-api-keys)\
&emsp;[Caching](#caching)\
&emsp;[Token counter](#token-counter)\
&emsp;[Where Loom stores state](#where-loom-stores-state)\
[4. Contributing](#4-contributing)\
[5. License](#5-license)

## 2. Getting started

### Installation

```bash
pip install loom-batch
```

The PyPI package is `loom-batch` (the name `loom` was taken); the CLI command is `loom`.

To work on Loom itself, see [CONTRIBUTING.md](CONTRIBUTING.md).

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

**Parquet** — works exactly like CSV.

### Submit a batch request

Minimal end-to-end run, passing the API key inline (see [Storing API keys](#storing-api-keys) for cleaner options):

```bash
loom run --file prompts.json \
         --provider openai \
         --model gpt-5.4-mini \
         --api-key sk-...
# -> Batch submitted. id=batch_abc123 provider=openai

# ...minutes or hours later...
loom fetch              # --all is the default; fetches every pending batch
# -> Fabric complete. id=batch_abc123 -> prompts_results_openai_gpt-5.4-mini.json
```

The output is written next to the input as `<name>_results_<provider>_<model>.<ext>`. Forward slashes and other unsafe characters in the model id are replaced with underscores (e.g. `openai/gpt-5.4-mini` → `openai_gpt-5.4-mini`). For gzipped inputs the `.gz` is dropped — `data.csv.gz` → `data_results_<provider>_<model>.csv`. Override the path entirely with `--output`.

### Python library

```python
from loom import Loom

client = Loom("google", "gemini-3.5-flash", cache_dir="/tmp/loom-cache", temperature=0.2, max_tokens=500)

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
| `--provider`, `-p`   | _required_                                 | `openai`, `anthropic`, `google`, `openrouter`, or `alibaba`.                                                           |
| `--model`, `-m`      | _required_                                 | Provider-specific model id (e.g. `gpt-5.4-mini`, `claude-haiku-4-5`, `gemini-3.5-flash`, `openai/gpt-5.4-mini`, `qwen-plus`). |
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

For pending batches, `loom fetch` prints the current status (`validating`, `in_progress`, …) and picks the batch up again on the next run. All statuses are explained in [docs/batch-jobs.md](docs/batch-jobs.md#batch-statuses).

#### `loom list`

List every batch known to Loom, with last-seen status, model, and source file. No flags.

#### `loom tokens`

Count input tokens for every prompt using the provider's token-counting API. See [Token counter](#token-counter).

| Flag               | Default      | Description                                       |
| ------------------ | ------------ | ------------------------------------------------- |
| `--file`, `-f`     | _required_   | Input `.json`, `.csv`, `.parquet`, `.json.gz`, or `.csv.gz`. |
| `--provider`, `-p` | _required_   | `openai`, `anthropic`, `google`, `openrouter`, or `alibaba`. |
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

| Setting             | CLI flag               | OpenAI                  | Anthropic        | Google (Gemini)         | OpenRouter, Alibaba |
| ------------------- | ---------------------- | ----------------------- | ---------------- | ----------------------- | ------------------- |
| `temperature`       | `--temperature`, `-t`  | `temperature`           | `temperature`    | `temperature`           | `temperature`       |
| `max_tokens`        | `--max-tokens`         | `max_completion_tokens` | `max_tokens` ¹   | `max_output_tokens`     | `max_tokens`        |
| `top_p`             | `--top-p`              | `top_p`                 | `top_p`          | `top_p`                 | `top_p`             |
| `top_k`             | `--top-k`              | ✗                       | `top_k`          | `top_k`                 | `top_k`             |
| `stop`              | `--stop` (repeatable)  | `stop`                  | `stop_sequences` | `stop_sequences`        | `stop`              |
| `seed`              | `--seed`               | `seed`                  | ✗                | `seed`                  | `seed`              |
| `presence_penalty`  | `--presence-penalty`   | `presence_penalty`      | ✗                | `presence_penalty`      | `presence_penalty`  |
| `frequency_penalty` | `--frequency-penalty`  | `frequency_penalty`     | ✗                | `frequency_penalty`     | `frequency_penalty` |
| `system`            | `--system`             | system message          | `system`         | `system_instruction`    | system message      |
| `json_mode`         | `--json`               | `response_format` JSON  | ✗                | `response_mime_type`    | `response_format`   |
| `extra`             | `--param key=value`    | request body            | message params   | `GenerateContentConfig` | request body        |

¹ Anthropic requires `max_tokens`; Loom sends 4096 when it is not set.

A setting marked ✗ raises `UnsupportedParameterError` before anything is sent — Loom never drops a setting silently.
Unset settings are not sent, so the provider default applies. Batch and sync requests carry exactly the same settings.
`extra` passes provider-specific options through unchanged (e.g. `reasoning_effort` for OpenAI reasoning models,
`thinking_config` for Gemini); Loom does not validate them.

```bash
loom run -p google -m gemini-3.5-flash -f data.csv -t 0.2 --max-tokens 800 --system "Answer in German." --json
loom run -p anthropic -m claude-haiku-4-5 -f data.csv --temperature 0 --stop "###"
loom run -p openai -m gpt-5.4-mini -f data.csv --param reasoning_effort=low
```

```python
from loom import Loom, GenerationParams

client = Loom("google", "gemini-3.5-flash", temperature=0.2, max_tokens=800)
client.generate("…", params={"temperature": 0.9})        # override for one call
client = Loom("openai", "gpt-5.4-mini", params=GenerationParams(seed=7, json_mode=True))
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
DASHSCOPE_API_KEY=sk-...
# optional, Alibaba Cloud: region/workspace endpoint (default Singapore)
# DASHSCOPE_BASE_URL=https://dashscope-intl.aliyuncs.com/compatible-mode/v1
```

**Alibaba Cloud (Model Studio).** Batch and sync use the OpenAI-compatible API. API keys, endpoints and model lists are
per region; set `DASHSCOPE_BASE_URL` to the endpoint of the key's region (Beijing:
`https://dashscope.aliyuncs.com/compatible-mode/v1`, workspace endpoints:
`https://{WorkspaceId}.{region}.maas.aliyuncs.com/compatible-mode/v1`). A batch file holds one model and one
thinking mode; pass switches like `enable_thinking` via `extra`.

A `.env` in the working directory is the friction-free option for daily use; `--api-key` is handy for one-offs or shared workstations.

### Caching

Loom caches every response under `~/.loom/cache/` (override with `--cache-dir`, `$LOOM_CACHE_DIR`, or `$LOOM_HOME`). The cache key is `sha256("<provider>|<model>|<prompt>|<settings>")`, so changing any of those misses the cache. `<settings>` is the canonical JSON of the [generation settings](#generation-settings) that are set; with no settings it is left out, so caches from Loom ≤ 0.4 stay valid. There is no TTL or eviction — the cache grows monotonically until you clear it.

**Sync mode** reads the cache before calling the provider and writes every successful response.

**Batch mode** also uses the cache:

- at submit time, cached prompts are skipped (only misses go to the provider);
- if *every* prompt is cached, Loom writes the output immediately and skips the provider entirely;
- at fetch time, newly downloaded responses are written into the cache.

```bash
loom run --sync -p openai -m gpt-5.4-mini -f data.csv -c text   # first run: API calls
loom run --sync -p openai -m gpt-5.4-mini -f data.csv -c text   # second run: 100% cache hits
loom run -p openai -m gpt-5.4-mini -f data.csv -c text          # batch: skips cached prompts
loom run --sync -p openai -m gpt-5.4-mini -f data.csv --no-cache
loom run --sync -p openai -m gpt-5.4-mini -f data.csv --cache-dir /tmp/my-cache
loom cache clear                                                # wipe the cache directory
loom cache clear --cache-dir /tmp/my-cache
```

`loom run --sync` reports cache hits live in its progress bar. From Python, pass `cache_dir=` to [`Loom`](docs/api.md#cache-configuration).

### Token counter

```bash
loom tokens --file prompts.json --provider anthropic --model claude-haiku-4-5
# Counting tokens ████████░░░░  340/1000  est_total≈36,210  errors=0  0:01:12  eta 0:02:35
# -> Total input tokens: 12,345 across 100 prompts (provider=anthropic, model=claude-haiku-4-5, errors=0)
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
| OpenAI     | `client.responses.input_tokens.count(...)` → `input_tokens` | ✓                                  |
| OpenRouter | —                                                    | ✗                                         |
| Alibaba Cloud | —                                                 | ✗                                         |

For unsupported providers, `loom tokens` prints _"Token counting not available: ..."_ and exits with code 2.

### Where Loom stores state

Everything lives under `~/.loom/` (override with `$LOOM_HOME`): pending batch jobs in `batches/`, the response cache in `cache/`, and prompt snapshots for batches submitted from Python in `inputs/`. The cache is safe to delete; deleting `batches/` makes Loom lose track of batches still running. Details in [docs/batch-jobs.md](docs/batch-jobs.md#where-loom-stores-state).

## 4. Contributing

Setup from source, repository layout, tests, provider evaluations, and the release process are in [CONTRIBUTING.md](CONTRIBUTING.md).

## 5. License

MIT — see [LICENSE](LICENSE).
