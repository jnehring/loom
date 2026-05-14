# loom

A small command-line tool for **LLM batch processing** across OpenAI, Anthropic, and Google Gemini.

Batch APIs let you submit a large number of prompts in one go, then come back later (up to 24h) to collect the results — typically at ~50% the price of synchronous calls. `loom` gives you a single interface over all three providers:

- Submit a `.json` or `.csv` file full of prompts.
- The tool converts it to the provider's batch format and submits it.
- Batch metadata is persisted under `~/.loom/batches/` so you can come back later.
- Fetch results when ready; `loom` merges them back into a `_results.json` or `_results.csv` next to your input, preserving original columns and IDs.

Provider SDKs are **not** used — `loom` talks to the batch HTTP APIs directly via `httpx`. Five dependencies total.

## Installation

Requires Python 3.10+.

```bash
git clone <this-repo> loom
cd loom
uv venv
uv pip install -e .
```

Or with plain `pip`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### API keys

`loom` reads keys, in order of precedence, from:

1. The `--api-key` CLI flag
2. Environment variables: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GOOGLE_API_KEY`
3. A `.env` file in the current directory

## Usage

### Input formats

**JSON** — a list of `{id, prompt}` objects:

```json
[
  {"id": "task-001", "prompt": "Summarize this: ..."},
  {"id": "task-002", "prompt": "Translate to French: ..."}
]
```

**CSV** — any schema; specify which column holds the prompt with `--col`:

```csv
id,prompt
1,"Explain quantum physics"
2,"Write a poem about rust"
```

### Sending a batch

```bash
# JSON input with OpenAI
loom run --file prompts.json --provider openai --model gpt-4o-mini

# CSV input with Anthropic
loom run --file data.csv --col prompt --provider anthropic --model claude-3-5-sonnet-latest

# Gemini
loom run --file prompts.json --provider google --model gemini-2.5-flash
```

`loom` prints the assigned `batch_id` and saves state to `~/.loom/batches/{provider}_{batch_id}.json`. The batch then runs asynchronously on the provider's side — you can close your terminal.

### Retrieving results

```bash
# Fetch a specific batch
loom fetch --id batch_67890abc

# Fetch every batch tracked locally
loom fetch --all
```

If the batch is still running, `loom` reports progress. When complete, results are written next to the original input file:

- `prompts.json` → `prompts_results.json` (each row gains an `llm_response` field)
- `data.csv` → `data_results.csv` (all original columns preserved, `llm_response` appended)

### Other commands

```bash
loom list           # show every batch tracked locally
loom forget <id>    # drop a batch from local state (does not cancel remotely)
```

## Supported providers

| Provider  | Env var             | Example model                  |
|-----------|---------------------|--------------------------------|
| OpenAI    | `OPENAI_API_KEY`    | `gpt-4o-mini`                  |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-3-5-sonnet-latest`     |
| Google    | `GOOGLE_API_KEY`    | `gemini-2.5-flash`             |
