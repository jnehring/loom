# Loom Python API

Loom can be used as a Python library in addition to the CLI. Install the same package:

```bash
pip install loom-batch
```

```python
from loom import Loom

client = Loom("openai", "gpt-4o-mini")
print(client.generate("Say hello in one word."))
```

API keys are resolved the same way as the CLI: explicit `api_key=` argument → environment variable → `.env` file. See the [README](../README.md#storing-api-keys).

---

## Quickstart

### Single prompt

```python
from loom import Loom, generate

# Client (reusable)
client = Loom("anthropic", "claude-3-5-sonnet-latest")
text = client.generate("Summarize Hamlet in one sentence.")

# One-liner
text = generate("Summarize Hamlet in one sentence.", "openai", "gpt-4o-mini")
```

### Many prompts

```python
result = client.generate_many([
    "Translate 'hello' to French.",
    "Translate 'hello' to German.",
])
print(result.texts)       # ["Bonjour", "Hallo"]
print(result.cache_hits)  # 0 on first run
print(result.errors)      # 0 if all succeeded

for r in result.results:
    if r.error:
        print(f"{r.id} failed: {r.error}")
    elif r.cached:
        print(f"{r.id} (cache): {r.text}")
```

### DataFrame column

```python
import pandas as pd

df = pd.DataFrame({"text": ["prompt a", "prompt b"]})
out = client.generate_frame(df, column="text")
# out has an added "llm_response" column
```

### File in → file out (sync)

```python
run = client.run_file("prompts.json", force=True)
print(run.output_path, run.cache_hits, run.errors)
```

### Batch submit → fetch

```python
job = client.submit_file("prompts.csv", column="text")
print(job.id, job.status)   # e.g. batch_abc… / validating

# later…
fetch = job.fetch()
if fetch.done:
    print("wrote", fetch.output_path)
```

In-memory batch:

```python
job = client.submit(["prompt a", "prompt b"])
fetch = job.fetch()
if fetch.done:
    print(fetch.responses)  # {"prompt-0": "...", "prompt-1": "..."}
```

---

## `Loom`

```python
Loom(
    provider: Literal["openai", "anthropic", "google", "openrouter"],
    model: str,
    *,
    api_key: str | None = None,
    workers: int = 8,
    use_cache: bool = True,
    cache_dir: str | Path | None = None,
    with_meta: bool = False,
)
```

| Parameter    | Description |
| ------------ | ----------- |
| `provider`   | Target LLM provider. OpenRouter only supports sync methods (no batch). |
| `model`      | Provider-specific model id. |
| `api_key`    | Override env / `.env` resolution for this client. |
| `workers`    | Concurrent workers for sync generation and token counting. |
| `use_cache`  | Read/write the on-disk response cache (default on). |
| `cache_dir`  | Override cache location (see [Cache configuration](#cache-configuration)). |
| `with_meta`  | Add `llm_provider` / `llm_model` to file and DataFrame outputs. |

### Properties

- `cache` → [`ResponseCache`](#responsecache) used by this client.

### In-memory methods

#### `generate(prompt: str) -> str`

Generate a single response. Raises `RuntimeError` if the provider call fails.

#### `generate_many(prompts, *, on_progress=None) -> GenerationResult`

Generate many responses concurrently. Per-prompt errors are captured on
`PromptResult.error` rather than raising. `on_progress(done, total, cache_hits, errors)`
is called after each prompt completes.

#### `generate_frame(df, column="text", *, on_progress=None) -> DataFrame`

Run every value in `column` and return a copy with an `llm_response` column
(plus meta columns when `with_meta=True`).

#### `count_tokens(prompts, *, on_progress=None) -> TokenCountResult`

Count input tokens for one string or a sequence of strings. Raises
`TokenCountingNotSupported` for providers without a remote counting API
(OpenAI, OpenRouter).

### File-based methods

#### `run_file(path, *, column="text", output=None, force=False, on_progress=None) -> RunResult`

Synchronously process a JSON / CSV / Parquet file and write the merged output.
Raises `SyncOutputExistsError` if the output exists and `force=False`.

#### `submit_file(path, *, column="text", output=None, force=False) -> BatchJob`

Submit a file as a provider batch job. Cached prompts are skipped at submit
time; a fully-cached dataset returns a completed local job immediately
(no provider call). Raises `OutputExistsError` in that fully-cached case when
the output exists and `force=False`.

#### `submit(prompts) -> BatchJob`

Submit an in-memory list of prompts as a batch job. Prompt snapshots are stored
under `~/.loom/inputs/` so `fetch` can rebuild cache keys later.

### Batch lifecycle

#### `job(batch_id) -> BatchJob`

Load a previously submitted batch by id.

#### `jobs() -> list[BatchJob]`

List every batch known to Loom (under `~/.loom/batches/`).

---

## Result types

### `PromptResult`

| Field    | Type         | Description |
| -------- | ------------ | ----------- |
| `id`     | `str`        | Internal id (`prompt-0`, … or original JSON id). |
| `prompt` | `str`        | Input prompt. |
| `text`   | `str`        | Response text (empty on error). |
| `error`  | `str \| None`| Error message if the provider call failed. |
| `cached` | `bool`       | True if served from the on-disk cache. |

### `GenerationResult`

| Field / property | Description |
| ---------------- | ----------- |
| `results`        | List of `PromptResult` in input order. |
| `total`          | Number of prompts. |
| `cache_hits`     | How many were served from cache. |
| `errors`         | How many failed. |
| `texts`          | Convenience: `[r.text for r in results]`. |
| `error_messages` | `{id: error}` for failed prompts. |

### `RunResult`

| Field            | Description |
| ---------------- | ----------- |
| `output_path`    | Path written. |
| `total`          | Number of prompts. |
| `cache_hits`     | Cache hits. |
| `errors`         | Failures. |
| `error_messages` | `{id: error}` map. |

### `TokenCountResult`

| Field          | Description |
| -------------- | ----------- |
| `total_tokens` | Sum of successfully counted input tokens. |
| `total`        | Number of prompts. |
| `errors`       | Failures. |

### `BatchFetchResult`

| Field            | Description |
| ---------------- | ----------- |
| `job`            | Updated `BatchJob`. |
| `done`           | True when results are available. |
| `responses`      | `{custom_id: text}` (always filled when done). |
| `error_messages` | Per-prompt or batch-level errors. |
| `output_path`    | Merged file path (file-sourced jobs only). |

---

## `BatchJob`

| Property / method | Description |
| ----------------- | ----------- |
| `id`              | Provider batch id (or `local-…` for fully-cached jobs). |
| `status`          | Last-known status string. |
| `is_done`         | `status == "completed"`. |
| `provider` / `model` | As submitted. |
| `output_path`     | Merged output path if known. |
| `meta`            | Underlying `BatchMetadata`. |
| `from_cache`      | True when the job was served entirely from cache. |
| `refresh()`       | Poll the provider; returns the new status. |
| `fetch(*, force=False, keep=False)` | Download/merge results → `BatchFetchResult`. |

---

## Module-level convenience

```python
from loom import generate, generate_many, run_file

generate(prompt, provider, model, *, api_key=None, use_cache=True, cache_dir=None) -> str

generate_many(prompts, provider, model, *, api_key=None, workers=8,
              use_cache=True, cache_dir=None, on_progress=None) -> GenerationResult

run_file(path, provider, model, *, column="text", api_key=None, output=None,
         workers=8, use_cache=True, cache_dir=None, force=False,
         with_meta=False, on_progress=None) -> RunResult
```

Each constructs a throwaway `Loom` client.

---

## Cache configuration

### Location resolution

1. Explicit `cache_dir=` argument on `Loom` / `ResponseCache` / CLI `--cache-dir`
2. `$LOOM_CACHE_DIR`
3. `$LOOM_HOME/cache` (default `$LOOM_HOME` is `~/.loom`)

Batch metadata lives under `$LOOM_HOME/batches/`; in-memory batch prompt snapshots under `$LOOM_HOME/inputs/`.

```python
client = Loom("openai", "gpt-4o-mini", cache_dir="/tmp/my-loom-cache")
# or
import os
os.environ["LOOM_CACHE_DIR"] = "/tmp/my-loom-cache"
os.environ["LOOM_HOME"] = "/tmp/loom-state"
```

### Cache key

```
sha256("<provider>" + "\0" + "<model>" + "\0" + "<prompt>")
```

Changing any of the three produces a miss. There is no TTL or eviction.

### `ResponseCache`

```python
from loom import ResponseCache

cache = ResponseCache(cache_dir="/tmp/c")
cache.get("openai", "gpt-4o-mini", "hello")   # str | None
cache.set("openai", "gpt-4o-mini", "hello", "world")
cache.count()   # int
cache.clear()   # returns number of files removed
cache.dir       # Path
cache.enabled   # bool
```

Pass `enabled=False` for a no-op cache (equivalent to `use_cache=False` on `Loom`).

### Batch-mode caching

Batch mode uses the same cache as sync:

- **Submit:** cached prompts are skipped; only misses go to the provider. Hits are stored on the batch metadata and merged back at fetch time.
- **Fully cached:** if every prompt hits, Loom writes the output immediately (file-sourced) / returns cached responses (in-memory) with a synthetic `local-…` batch id — no provider call, nothing persisted under `batches/`.
- **Fetch:** newly downloaded responses are written into the cache so a later sync or batch run of the same prompts is free.

Disable with `use_cache=False` or CLI `--no-cache`.

---

## Exceptions

| Exception | When |
| --------- | ---- |
| `OutputExistsError` | File-based batch merge (or fully-cached submit) would overwrite an existing file and `force=False`. Has `.meta` and `.out_path`. |
| `SyncOutputExistsError` | `run_file` / `generate_sync` would overwrite and `force=False`. Has `.out_path`. |
| `TokenCountingNotSupported` | Provider has no remote token-counting API. |
| `FileNotFoundError` | Input path missing, or unknown batch id. |
| `RuntimeError` | Missing API key, or `generate()` hit a provider error. |
| `ValueError` | Empty input, unknown provider, missing CSV column, etc. |

---

## Concurrency and thread safety

- Sync generation and token counting use a `ThreadPoolExecutor` sized by `workers`.
- The on-disk cache writes one file per entry keyed by content hash; concurrent writers to *different* keys are safe. Concurrent writes to the *same* key may race but produce identical content.
- Do not share a single `Loom` instance across processes; each process should construct its own client (the cache directory can be shared).

---

## CLI parity

| Library | CLI |
| ------- | --- |
| `client.run_file(...)` | `loom run --sync -f ...` |
| `client.submit_file(...)` | `loom run -f ...` (batch) |
| `job.fetch()` | `loom fetch --id ...` |
| `client.jobs()` | `loom list` |
| `client.count_tokens(...)` on a file | `loom tokens -f ...` |
| `client.cache.clear()` | `loom cache clear` |
