"""Top-level orchestration: convert input, submit, persist, fetch, merge."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Optional

from ..core.models import BatchMetadata, ProviderName, PromptItem
from ..providers import get_provider, get_sync_provider
from ..utils import cache as response_cache
from ..utils import converters, storage
from ..utils.keys import resolve_api_key


def run_batch(
    file_path: Path,
    provider_name: ProviderName,
    model: str,
    column: Optional[str] = None,
    api_key: Optional[str] = None,
    output_path: Optional[Path] = None,
    with_meta: bool = False,
) -> BatchMetadata:
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    file_type, _is_gz = converters.detect_format(file_path)
    if file_type == "csv":
        if not column:
            raise ValueError("--col is required for CSV input.")
        items, _df, id_map = converters.load_csv(file_path, column)
        prompt_column = column
    else:  # json
        items, id_map = converters.load_json(file_path)
        prompt_column = None

    if not items:
        raise ValueError("Input file contains no prompts.")

    key = resolve_api_key(provider_name, api_key)
    provider = get_provider(provider_name, key)
    batch_id = provider.submit(items, model)

    meta = BatchMetadata(
        batch_id=batch_id,
        provider=provider_name,
        model=model,
        original_file_path=str(file_path),
        file_type=file_type,
        prompt_column=prompt_column,
        id_map=id_map,
        status="validating",
        output_path=str(output_path) if output_path else None,
        with_meta=with_meta,
    )
    storage.save_batch(meta)
    return meta


class OutputExistsError(Exception):
    """Raised when the merged output file would overwrite an existing file."""

    def __init__(self, meta: BatchMetadata, out_path: Path) -> None:
        super().__init__(f"Output file already exists: {out_path}")
        self.meta = meta
        self.out_path = out_path


def fetch_batch(
    batch_id: str,
    api_key: Optional[str] = None,
    keep: bool = False,
    force: bool = False,
) -> tuple[BatchMetadata, bool]:
    """Return (metadata, done). If done, results have been merged to output_path.

    On successful completion the stored metadata file in ~/.loom/batches/ is
    deleted, unless ``keep=True`` is passed.

    If the target output file already exists and ``force=False``, raises
    :class:`OutputExistsError` BEFORE downloading results — the caller (CLI)
    can prompt the user and retry with ``force=True``.
    """
    meta = storage.load_batch(batch_id)
    key = resolve_api_key(meta.provider, api_key)
    provider = get_provider(meta.provider, key)
    status = provider.check_status(meta.batch_id)
    meta.status = status

    if status != "completed":
        storage.save_batch(meta)
        return meta, False

    original = Path(meta.original_file_path)
    out_path = (
        Path(meta.output_path)
        if meta.output_path
        else converters.default_output_path(original, meta.provider, meta.model)
    )

    if out_path.exists() and not force:
        raise OutputExistsError(meta, out_path)

    responses = provider.download_results(meta.batch_id)

    if meta.file_type == "json":
        converters.merge_json(
            original, responses, out_path,
            with_meta=meta.with_meta, provider=meta.provider, model=meta.model,
        )
    else:
        converters.merge_csv(
            original, meta.id_map, responses, out_path,
            with_meta=meta.with_meta, provider=meta.provider, model=meta.model,
        )

    meta.output_path = str(out_path)

    if keep:
        storage.save_batch(meta)
    else:
        storage.delete_batch(meta.batch_id)
    return meta, True


def fetch_all(
    api_key: Optional[str] = None,
    keep: bool = False,
    force: bool = False,
    on_conflict=None,
) -> list[tuple[BatchMetadata, bool]]:
    """Fetch every pending batch.

    ``on_conflict`` is an optional callable ``(meta, out_path) -> bool`` invoked
    when the merged output file already exists. Return True to overwrite,
    False to skip. If not provided and ``force=False``, conflicts are skipped.
    """
    results = []
    for meta in storage.list_batches():
        if meta.status in {"completed", "failed", "expired", "cancelled"} and meta.output_path:
            continue
        try:
            results.append(
                fetch_batch(meta.batch_id, api_key=api_key, keep=keep, force=force)
            )
        except OutputExistsError as exc:
            overwrite = bool(on_conflict(exc.meta, exc.out_path)) if on_conflict else False
            if overwrite:
                results.append(
                    fetch_batch(meta.batch_id, api_key=api_key, keep=keep, force=True)
                )
            else:
                results.append((exc.meta, False))
        except Exception:  # noqa: BLE001
            results.append((meta, False))
            meta.status = "unknown"
    return results


# ---------- Synchronous (non-batch) generation ----------

class SyncOutputExistsError(Exception):
    """Raised when ``generate_sync`` would overwrite an existing output file."""

    def __init__(self, out_path: Path) -> None:
        super().__init__(f"Output file already exists: {out_path}")
        self.out_path = out_path


def generate_sync(
    file_path: Path,
    provider_name: ProviderName,
    model: str,
    column: Optional[str] = None,
    api_key: Optional[str] = None,
    output_path: Optional[Path] = None,
    workers: int = 8,
    use_cache: bool = True,
    force: bool = False,
    with_meta: bool = False,
    on_progress: Optional[Callable[[int, int, int, int], None]] = None,
) -> tuple[Path, int, int, int]:
    """Run prompts synchronously through a non-batch provider and write output.

    Returns ``(output_path, total, cache_hits, errors)``.

    ``on_progress`` is called after each prompt completes with
    ``(done, total, cache_hits, errors)``.
    """
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    file_type, _is_gz = converters.detect_format(file_path)
    if file_type == "csv":
        if not column:
            raise ValueError("--col is required for CSV input.")
        items, _df, id_map = converters.load_csv(file_path, column)
    else:  # json
        items, id_map = converters.load_json(file_path)

    if not items:
        raise ValueError("Input file contains no prompts.")

    out_path = (
        Path(output_path)
        if output_path
        else converters.default_output_path(file_path, provider_name, model)
    )
    if out_path.exists() and not force:
        raise SyncOutputExistsError(out_path)

    key = resolve_api_key(provider_name, api_key)
    provider = get_sync_provider(provider_name, key)

    total = len(items)
    responses: dict[str, str] = {}
    cache_hits = 0
    errors = 0

    # Separate cache hits from items that need to be fetched.
    pending: list[PromptItem] = []
    if use_cache:
        for it in items:
            cached = response_cache.get(provider_name, model, it.prompt)
            if cached is not None:
                responses[it.custom_id] = cached
                cache_hits += 1
            else:
                pending.append(it)
    else:
        pending = list(items)

    done = cache_hits
    if on_progress:
        on_progress(done, total, cache_hits, errors)

    def _run_one(item: PromptItem) -> tuple[str, str, Optional[Exception]]:
        try:
            text = provider.generate(item.prompt, model)
            if use_cache:
                response_cache.set(provider_name, model, item.prompt, text)
            return item.custom_id, text, None
        except Exception as exc:  # noqa: BLE001
            return item.custom_id, "", exc

    if pending:
        n_workers = max(1, min(workers, len(pending)))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_run_one, it) for it in pending]
            for fut in as_completed(futures):
                cid, text, err = fut.result()
                responses[cid] = text
                if err is not None:
                    errors += 1
                done += 1
                if on_progress:
                    on_progress(done, total, cache_hits, errors)

    if file_type == "json":
        converters.merge_json(
            file_path, responses, out_path,
            with_meta=with_meta, provider=provider_name, model=model,
        )
    else:
        converters.merge_csv(
            file_path, id_map, responses, out_path,
            with_meta=with_meta, provider=provider_name, model=model,
        )

    return out_path, total, cache_hits, errors


# ---------- Token counting ----------

class TokenCountingNotSupported(Exception):
    """Raised when a provider does not expose a token-counting API."""


def count_tokens(
    file_path: Path,
    provider_name: ProviderName,
    model: str,
    column: Optional[str] = None,
    api_key: Optional[str] = None,
    workers: int = 8,
    on_progress: Optional[Callable[[int, int, int, int], None]] = None,
) -> tuple[int, int, int]:
    """Count input tokens for every prompt in the input file.

    Returns ``(total_tokens, total_prompts, errors)``.

    Raises :class:`TokenCountingNotSupported` if the provider has no
    token-counting API.

    ``on_progress`` is called after each prompt with
    ``(done, total, errors, tokens_so_far)`` where ``tokens_so_far`` is the
    running sum of successfully-counted prompt tokens (used by callers to
    project a final-total estimate).
    """
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    file_type, _is_gz = converters.detect_format(file_path)
    if file_type == "csv":
        if not column:
            raise ValueError("--col is required for CSV input.")
        items, _df, _id_map = converters.load_csv(file_path, column)
    else:  # json
        items, _id_map = converters.load_json(file_path)

    if not items:
        raise ValueError("Input file contains no prompts.")

    key = resolve_api_key(provider_name, api_key)
    provider = get_sync_provider(provider_name, key)

    # Probe support cheaply with the first prompt before fanning out.
    probe = provider.count_tokens(items[0].prompt, model)
    if probe is None:
        raise TokenCountingNotSupported(
            f"Provider '{provider_name}' does not expose a token-counting API."
        )

    total = len(items)
    total_tokens = int(probe)
    errors = 0
    done = 1
    if on_progress:
        on_progress(done, total, errors, total_tokens)

    def _run_one(item: PromptItem) -> tuple[Optional[int], Optional[Exception]]:
        try:
            n = provider.count_tokens(item.prompt, model)
            return (int(n) if n is not None else None, None)
        except Exception as exc:  # noqa: BLE001
            return None, exc

    remaining = items[1:]
    if remaining:
        n_workers = max(1, min(workers, len(remaining)))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_run_one, it) for it in remaining]
            for fut in as_completed(futures):
                n, err = fut.result()
                if err is not None or n is None:
                    errors += 1
                else:
                    total_tokens += n
                done += 1
                if on_progress:
                    on_progress(done, total, errors, total_tokens)

    return total_tokens, total, errors
