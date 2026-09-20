"""Top-level orchestration: convert input, submit, persist, fetch, merge."""

from __future__ import annotations

import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

from ..core.models import BatchMetadata, ProviderName, PromptItem
from ..providers import get_provider, get_sync_provider
from ..utils import cache as response_cache
from ..utils import converters, storage
from ..utils.cache import ResponseCache
from ..utils.errors import format_exception
from ..utils.keys import resolve_api_key

PathLike = Union[str, Path]


# ---------- Shared helpers ----------

def _resolve_cache(
    *,
    use_cache: bool = True,
    cache_dir: Optional[PathLike] = None,
    cache: Optional[ResponseCache] = None,
) -> ResponseCache:
    if cache is not None:
        return cache
    return response_cache.get_cache(cache_dir=cache_dir, enabled=use_cache)


def _load_input(
    file_path: Path,
    file_type: str,
    column: str,
) -> tuple[list[PromptItem], dict[str, str], Optional[str]]:
    if file_type == "json":
        items, id_map = converters.load_json(file_path)
        return items, id_map, None
    if file_type in ("csv", "parquet"):
        if not column:
            raise ValueError(f"--col is required for {file_type.upper()} input.")
        loader = converters.load_csv if file_type == "csv" else converters.load_parquet
        items, _df, id_map = loader(file_path, column)
        return items, id_map, column
    raise ValueError(f"Unsupported file type: {file_type}")


def _merge_output(
    file_type: str,
    original: Path,
    id_map: dict[str, str],
    responses: dict[str, str],
    out_path: Path,
    *,
    with_meta: bool = False,
    provider: str | None = None,
    model: str | None = None,
) -> None:
    kwargs = {"with_meta": with_meta, "provider": provider, "model": model}
    if file_type == "json":
        converters.merge_json(original, responses, out_path, **kwargs)
    elif file_type == "csv":
        converters.merge_csv(original, id_map, responses, out_path, **kwargs)
    elif file_type == "parquet":
        converters.merge_parquet(original, id_map, responses, out_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")


def _prompts_by_custom_id(items: Sequence[PromptItem]) -> dict[str, str]:
    return {it.custom_id: it.prompt for it in items}


def _reload_prompts_for_batch(meta: BatchMetadata) -> dict[str, str]:
    """Reconstruct custom_id -> prompt for a stored batch (needed to populate cache)."""
    original = Path(meta.original_file_path)
    if not original.exists():
        return {}
    column = meta.prompt_column or "text"
    try:
        items, _id_map, _ = _load_input(original, meta.file_type, column)
    except Exception:  # noqa: BLE001
        return {}
    return _prompts_by_custom_id(items)


# ---------- Batch submission (items + file wrappers) ----------

class OutputExistsError(Exception):
    """Raised when the merged output file would overwrite an existing file."""

    def __init__(self, meta: BatchMetadata, out_path: Path) -> None:
        super().__init__(f"Output file already exists: {out_path}")
        self.meta = meta
        self.out_path = out_path


@dataclass
class SubmitResult:
    """Outcome of submitting items to a provider batch (or serving fully from cache)."""

    meta: BatchMetadata
    pending: list[PromptItem]
    from_cache: bool = False  # True when every prompt was a cache hit


def submit_items(
    items: Sequence[PromptItem],
    provider_name: ProviderName,
    model: str,
    *,
    api_key: Optional[str] = None,
    use_cache: bool = True,
    cache_dir: Optional[PathLike] = None,
    cache: Optional[ResponseCache] = None,
    id_map: Optional[dict[str, str]] = None,
    original_file_path: str = "",
    file_type: str = "json",
    prompt_column: Optional[str] = None,
    output_path: Optional[PathLike] = None,
    with_meta: bool = False,
    source: str = "file",
    persist: bool = True,
) -> SubmitResult:
    """Filter cache hits, submit the rest to the provider batch API.

    If every prompt is a cache hit, no provider call is made and the returned
    metadata has ``status="completed"`` with a synthetic ``local-...`` batch_id.
    """
    if not items:
        raise ValueError("No prompts to submit.")

    cache_obj = _resolve_cache(use_cache=use_cache, cache_dir=cache_dir, cache=cache)
    cached_responses: dict[str, str] = {}
    pending: list[PromptItem] = []
    for it in items:
        hit = cache_obj.get(provider_name, model, it.prompt)
        if hit is not None:
            cached_responses[it.custom_id] = hit
        else:
            pending.append(it)

    resolved_id_map = id_map if id_map is not None else {it.custom_id: it.custom_id for it in items}
    cache_dir_str = str(cache_obj.dir) if cache_obj.enabled else None

    if not pending:
        # Fully cached — fabricate a completed local batch (not persisted).
        batch_id = f"local-{uuid.uuid4().hex[:12]}"
        meta = BatchMetadata(
            batch_id=batch_id,
            provider=provider_name,
            model=model,
            original_file_path=original_file_path,
            file_type=file_type,  # type: ignore[arg-type]
            prompt_column=prompt_column,
            id_map=resolved_id_map,
            status="completed",
            output_path=str(output_path) if output_path else None,
            with_meta=with_meta,
            cached_responses=cached_responses,
            cache_dir=cache_dir_str,
            source=source,  # type: ignore[arg-type]
        )
        return SubmitResult(meta=meta, pending=[], from_cache=True)

    key = resolve_api_key(provider_name, api_key)
    provider = get_provider(provider_name, key)
    batch_id = provider.submit(list(pending), model)

    meta = BatchMetadata(
        batch_id=batch_id,
        provider=provider_name,
        model=model,
        original_file_path=original_file_path,
        file_type=file_type,  # type: ignore[arg-type]
        prompt_column=prompt_column,
        id_map=resolved_id_map,
        status="validating",
        output_path=str(output_path) if output_path else None,
        with_meta=with_meta,
        cached_responses=cached_responses,
        cache_dir=cache_dir_str,
        source=source,  # type: ignore[arg-type]
    )
    if persist:
        storage.save_batch(meta)
    return SubmitResult(meta=meta, pending=pending, from_cache=False)


def run_batch(
    file_path: Path,
    provider_name: ProviderName,
    model: str,
    column: str = "text",
    api_key: Optional[str] = None,
    output_path: Optional[Path] = None,
    with_meta: bool = False,
    use_cache: bool = True,
    cache_dir: Optional[PathLike] = None,
    force: bool = False,
) -> BatchMetadata:
    """Submit a file as a provider batch job (with optional cache filtering).

    When every prompt is already cached, merges the output immediately and
    returns a completed :class:`BatchMetadata` that was never persisted.
    """
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    file_type, _is_gz = converters.detect_format(file_path)
    items, id_map, prompt_column = _load_input(file_path, file_type, column)

    if not items:
        raise ValueError("Input file contains no prompts.")

    result = submit_items(
        items,
        provider_name,
        model,
        api_key=api_key,
        use_cache=use_cache,
        cache_dir=cache_dir,
        id_map=id_map,
        original_file_path=str(file_path),
        file_type=file_type,
        prompt_column=prompt_column,
        output_path=output_path,
        with_meta=with_meta,
        source="file",
        persist=True,
    )

    if result.from_cache:
        out_path = (
            Path(output_path)
            if output_path
            else converters.default_output_path(file_path, provider_name, model)
        )
        if out_path.exists() and not force:
            raise OutputExistsError(result.meta, out_path)
        _merge_output(
            file_type,
            file_path,
            id_map,
            result.meta.cached_responses,
            out_path,
            with_meta=with_meta,
            provider=provider_name,
            model=model,
        )
        result.meta.output_path = str(out_path)
        return result.meta

    return result.meta


def fetch_batch(
    batch_id: str,
    api_key: Optional[str] = None,
    keep: bool = False,
    force: bool = False,
    use_cache: bool = True,
) -> tuple[BatchMetadata, bool, dict[str, str], dict[str, str]]:
    """Return ``(metadata, done, prompt_errors, responses)``.

    If ``done``, file-sourced batches have been merged to ``output_path`` and
    ``responses`` maps ``custom_id -> text``. If not done, ``responses`` is empty.

    On successful completion the stored metadata file in ~/.loom/batches/ is
    deleted, unless ``keep=True`` is passed.

    If the target output file already exists and ``force=False``, raises
    :class:`OutputExistsError` BEFORE downloading results — the caller (CLI)
    can prompt the user and retry with ``force=True``.

    Newly downloaded responses are written into the response cache (unless
    ``use_cache=False``). Cached responses stored at submit time are merged in.
    """
    meta = storage.load_batch(batch_id)
    key = resolve_api_key(meta.provider, api_key)
    provider = get_provider(meta.provider, key)
    status = provider.check_status(meta.batch_id)
    meta.status = status

    if status != "completed":
        storage.save_batch(meta)
        batch_error = provider.batch_error_message(meta.batch_id)
        prompt_errors = {"batch": batch_error} if batch_error else {}
        return meta, False, prompt_errors, {}

    original = Path(meta.original_file_path)
    out_path = (
        Path(meta.output_path)
        if meta.output_path
        else converters.default_output_path(original, meta.provider, meta.model)
    )

    if meta.source == "file" and out_path.exists() and not force:
        raise OutputExistsError(meta, out_path)

    responses, prompt_errors = provider.download_results(meta.batch_id, id_map=meta.id_map)

    # Populate the cache with newly downloaded responses.
    cache_obj = _resolve_cache(
        use_cache=use_cache,
        cache_dir=meta.cache_dir,
    )
    if cache_obj.enabled and responses:
        prompts = _reload_prompts_for_batch(meta)
        for cid, text in responses.items():
            prompt = prompts.get(cid)
            if prompt is not None and text:
                cache_obj.set(meta.provider, meta.model, prompt, text)

    # Merge in responses that were served from cache at submit time.
    if meta.cached_responses:
        merged = dict(meta.cached_responses)
        merged.update(responses)
        responses = merged

    if meta.source == "memory":
        # In-memory batches: leave output writing to the library caller.
        meta.output_path = None
    else:
        _merge_output(
            meta.file_type,
            original,
            meta.id_map,
            responses,
            out_path,
            with_meta=meta.with_meta,
            provider=meta.provider,
            model=meta.model,
        )
        meta.output_path = str(out_path)

    if keep:
        storage.save_batch(meta)
    else:
        storage.delete_batch(meta.batch_id)
        if meta.source == "memory":
            storage.delete_memory_input(meta.provider, meta.batch_id)
    return meta, True, prompt_errors, responses


def fetch_all(
    api_key: Optional[str] = None,
    keep: bool = False,
    force: bool = False,
    on_conflict=None,
    use_cache: bool = True,
) -> list[tuple[BatchMetadata, bool, dict[str, str], dict[str, str]]]:
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
                fetch_batch(
                    meta.batch_id,
                    api_key=api_key,
                    keep=keep,
                    force=force,
                    use_cache=use_cache,
                )
            )
        except OutputExistsError as exc:
            overwrite = bool(on_conflict(exc.meta, exc.out_path)) if on_conflict else False
            if overwrite:
                results.append(
                    fetch_batch(
                        meta.batch_id,
                        api_key=api_key,
                        keep=keep,
                        force=True,
                        use_cache=use_cache,
                    )
                )
            else:
                results.append((exc.meta, False, {}, {}))
        except Exception:  # noqa: BLE001
            results.append((meta, False, {}, {}))
            meta.status = "unknown"
    return results


# ---------- Synchronous (non-batch) generation ----------

class SyncOutputExistsError(Exception):
    """Raised when ``generate_sync`` would overwrite an existing output file."""

    def __init__(self, out_path: Path) -> None:
        super().__init__(f"Output file already exists: {out_path}")
        self.out_path = out_path


@dataclass
class ItemResult:
    """Result for a single prompt in a sync run."""

    custom_id: str
    prompt: str
    text: str = ""
    error: Optional[str] = None
    cached: bool = False


@dataclass
class SyncRunResult:
    """Aggregate result of :func:`generate_items`."""

    results: list[ItemResult] = field(default_factory=list)
    total: int = 0
    cache_hits: int = 0
    errors: int = 0

    @property
    def responses(self) -> dict[str, str]:
        return {r.custom_id: r.text for r in self.results}

    @property
    def error_messages(self) -> dict[str, str]:
        return {r.custom_id: r.error for r in self.results if r.error}


def generate_items(
    items: Sequence[PromptItem],
    provider_name: ProviderName,
    model: str,
    *,
    api_key: Optional[str] = None,
    workers: int = 8,
    use_cache: bool = True,
    cache_dir: Optional[PathLike] = None,
    cache: Optional[ResponseCache] = None,
    on_progress: Optional[Callable[[int, int, int, int], None]] = None,
) -> SyncRunResult:
    """Run prompts synchronously through a non-batch provider.

    Returns a :class:`SyncRunResult`. ``on_progress`` is called after each
    prompt completes with ``(done, total, cache_hits, errors)``.
    """
    if not items:
        raise ValueError("No prompts to generate.")

    cache_obj = _resolve_cache(use_cache=use_cache, cache_dir=cache_dir, cache=cache)
    key = resolve_api_key(provider_name, api_key)
    provider = get_sync_provider(provider_name, key)

    total = len(items)
    by_id: dict[str, ItemResult] = {}
    cache_hits = 0
    errors = 0
    pending: list[PromptItem] = []

    for it in items:
        cached = cache_obj.get(provider_name, model, it.prompt)
        if cached is not None:
            by_id[it.custom_id] = ItemResult(
                custom_id=it.custom_id,
                prompt=it.prompt,
                text=cached,
                cached=True,
            )
            cache_hits += 1
        else:
            pending.append(it)

    done = cache_hits
    if on_progress:
        on_progress(done, total, cache_hits, errors)

    def _run_one(item: PromptItem) -> ItemResult:
        try:
            text = provider.generate(item.prompt, model)
            cache_obj.set(provider_name, model, item.prompt, text)
            return ItemResult(custom_id=item.custom_id, prompt=item.prompt, text=text)
        except Exception as exc:  # noqa: BLE001
            return ItemResult(
                custom_id=item.custom_id,
                prompt=item.prompt,
                text="",
                error=format_exception(exc),
            )

    if pending:
        n_workers = max(1, min(workers, len(pending)))
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_run_one, it) for it in pending]
            for fut in as_completed(futures):
                result = fut.result()
                by_id[result.custom_id] = result
                if result.error is not None:
                    errors += 1
                done += 1
                if on_progress:
                    on_progress(done, total, cache_hits, errors)

    # Preserve input order.
    ordered = [by_id[it.custom_id] for it in items]
    return SyncRunResult(
        results=ordered,
        total=total,
        cache_hits=cache_hits,
        errors=errors,
    )


def generate_sync(
    file_path: Path,
    provider_name: ProviderName,
    model: str,
    column: str = "text",
    api_key: Optional[str] = None,
    output_path: Optional[Path] = None,
    workers: int = 8,
    use_cache: bool = True,
    force: bool = False,
    with_meta: bool = False,
    cache_dir: Optional[PathLike] = None,
    on_progress: Optional[Callable[[int, int, int, int], None]] = None,
) -> tuple[Path, int, int, int, dict[str, str]]:
    """Run prompts synchronously through a non-batch provider and write output.

    Returns ``(output_path, total, cache_hits, errors, error_messages)``.

    ``on_progress`` is called after each prompt completes with
    ``(done, total, cache_hits, errors)``.
    """
    file_path = Path(file_path).resolve()
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    file_type, _is_gz = converters.detect_format(file_path)
    items, id_map, _prompt_column = _load_input(file_path, file_type, column)

    if not items:
        raise ValueError("Input file contains no prompts.")

    out_path = (
        Path(output_path)
        if output_path
        else converters.default_output_path(file_path, provider_name, model)
    )
    if out_path.exists() and not force:
        raise SyncOutputExistsError(out_path)

    run = generate_items(
        items,
        provider_name,
        model,
        api_key=api_key,
        workers=workers,
        use_cache=use_cache,
        cache_dir=cache_dir,
        on_progress=on_progress,
    )

    _merge_output(
        file_type,
        file_path,
        id_map,
        run.responses,
        out_path,
        with_meta=with_meta,
        provider=provider_name,
        model=model,
    )

    return out_path, run.total, run.cache_hits, run.errors, run.error_messages


# ---------- Token counting ----------

class TokenCountingNotSupported(Exception):
    """Raised when a provider does not expose a token-counting API."""


def count_token_items(
    items: Sequence[PromptItem],
    provider_name: ProviderName,
    model: str,
    *,
    api_key: Optional[str] = None,
    workers: int = 8,
    on_progress: Optional[Callable[[int, int, int, int], None]] = None,
) -> tuple[int, int, int]:
    """Count input tokens for a list of prompts.

    Returns ``(total_tokens, total_prompts, errors)``.

    Raises :class:`TokenCountingNotSupported` if the provider has no
    token-counting API.

    ``on_progress`` is called after each prompt with
    ``(done, total, errors, tokens_so_far)``.
    """
    if not items:
        raise ValueError("No prompts to count.")

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

    remaining = list(items[1:])
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


def count_tokens(
    file_path: Path,
    provider_name: ProviderName,
    model: str,
    column: str = "text",
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
    items, _id_map, _prompt_column = _load_input(file_path, file_type, column)

    if not items:
        raise ValueError("Input file contains no prompts.")

    return count_token_items(
        items,
        provider_name,
        model,
        api_key=api_key,
        workers=workers,
        on_progress=on_progress,
    )
