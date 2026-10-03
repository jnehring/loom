# Batch jobs: statuses and stored state

Details on what `loom fetch` reports and what Loom keeps on disk. For everyday use, the [README](../README.md) is enough.

## Batch statuses

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

## Where Loom stores state

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
