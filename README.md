# Loom

> Weave batch LLM jobs across OpenAI, Anthropic, and Google.

Loom is a small Python CLI for **batch LLM processing** of text-only prompts.
You hand it a JSON or CSV file of prompts, pick a provider and a model, and it
submits a batch job. Hours later you call `loom fetch`, and it merges the
responses back into the original file format.

Because batch jobs can take up to 24 hours, Loom acts as a tiny state machine:
batch IDs and metadata are persisted under `~/.loom/batches/`, so you can close
your terminal and come back tomorrow.

## Install

```bash
pip install loom-batch
```

The PyPI package is `loom-batch` (the name `loom` was taken), but the CLI command is `loom`.

For development:

```bash
git clone https://github.com/jannehring/loom
cd loom
pip install -e ".[dev]"
pytest
```

## Quick start

### Help

`loom` with no args prints help. So do all of these:

```bash
loom -h
loom --help
loom -?
loom run --help
loom fetch -h
```

### Setup

Create a `.env` (or export environment variables) with the keys you'll use:

```ini
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

Key resolution precedence: `--api-key` flag → environment variable → `.env`.

### JSON input

`prompts.json`:

```json
[
  {"id": "task-001", "prompt": "Summarize the plot of Hamlet in one sentence."},
  {"id": "task-002", "prompt": "Translate 'Good morning' to French."}
]
```

```bash
loom run --file prompts.json --provider openai --model gpt-4o-mini
# -> Batch submitted. id=batch_abc123 provider=openai

# ...later (minutes or hours)...
loom fetch --id batch_abc123
# -> Fabric complete. id=batch_abc123 -> prompts_results.json
```

`prompts_results.json` is the original file with an `llm_response` field added to each entry.

### CSV input

`data.csv`:

```csv
id,text,metadata
1,"Explain quantum physics in one paragraph",low_priority
2,"Write a haiku about rust",high_priority
```

```bash
loom run --file data.csv --col text --provider anthropic --model claude-3-5-sonnet-latest
loom fetch --all
```

The output `data_results.csv` preserves every original column and appends a new `llm_response` column. Row order is preserved.

## Commands

| Command | Purpose |
| --- | --- |
| `loom run` | Submit a new batch from a `.json` or `.csv` file. |
| `loom fetch --id <id>` | Check status / download results for one batch. Deletes the metadata file in `~/.loom/batches/` on success. |
| `loom fetch --all` | Process every pending batch in `~/.loom/batches/`. |
| `loom fetch ... --keep` | Same as above but keeps the metadata file after success. |
| `loom fetch ... --force` | Overwrite an existing output file without prompting (default: warn & confirm). |
| `loom list` | Show all known batches and their last-seen status. |
| `loom -h` / `-?` / `--help` | Show help. |

### `loom run` flags

| Flag | Description |
| --- | --- |
| `--file, -f` | Input file (required). |
| `--provider, -p` | One of `openai`, `anthropic`, `google`. |
| `--model, -m` | Provider-specific model id. |
| `--col, -c` | Prompt column (required for CSV). |
| `--api-key` | Override the env/.env key. |
| `--output, -o` | Custom output path (default: `<input>_results.<ext>`). |

## How it works

1. **Submit.** Loom converts your file into the provider's required JSONL/inline format, attaches a `custom_id` to every row, uploads, and creates the batch.
2. **Persist.** A small JSON file is saved to `~/.loom/batches/<provider>_<batch_id>.json`. It contains the batch id, original file path, file type, the prompt column (CSV), and a map from `custom_id` back to the original row key.
3. **Fetch.** `loom fetch` polls the provider. If the batch is done, it downloads results, looks up each `custom_id` in the id-map, and **merges responses back in original order**.

Providers don't guarantee response order — Loom always uses `custom_id` to put rows back where they belong.

## Provider matrix

| Feature | OpenAI | Anthropic | Google (Gemini) |
| --- | --- | --- | --- |
| Endpoint | `/v1/batches` | `/v1/messages/batches` | `batches.create` (genai SDK) |
| Input form | JSONL (uploaded) | inline `requests[]` | inline `src` list |
| Timeout | 24h | 24h | varies |
| Pricing | 50% off | 50% off | standard |

For very large Google batches the SDK supports a file/GCS upload path — Loom v0.1 only uses inline requests. If your dataset is huge, split it or fall back to OpenAI/Anthropic.

## Limitations (v0.1)

- Text in, text out. No images / tool use / structured output yet.
- Google: inline requests only (no GCS upload).
- One model per batch.
- No automatic retry on failed individual requests; failed items get an empty `llm_response`.

## Releasing

CI is wired up in `.github/workflows/`:

- **`test.yml`** runs `pytest` on every push & PR against Python 3.10 / 3.11 / 3.12.
- **`publish.yml`** runs tests, builds an sdist + wheel, and publishes to PyPI when you push a `v*` tag or publish a GitHub Release. It uses **PyPI Trusted Publishing (OIDC)** — no API token in secrets.

### One-time PyPI setup

1. Create the project on PyPI: https://pypi.org/manage/account/publishing/
2. Add a **trusted publisher** with these values:
   - Owner: `jannehring`
   - Repository: `loom`
   - Workflow: `publish.yml`
   - Environment: `pypi`
3. In GitHub: **Settings → Environments → New environment → `pypi`** (you can require manual approval here for extra safety).

### Cutting a release

```bash
# 1. Bump the version in pyproject.toml
# 2. Commit
git commit -am "Release v0.1.1"
# 3. Tag and push
git tag v0.1.1
git push origin main --tags
```

The `publish` workflow fires on the tag, runs tests, builds, and uploads to PyPI. You can also trigger it manually from the Actions tab (`workflow_dispatch`).

## License

MIT — see [LICENSE](LICENSE).
