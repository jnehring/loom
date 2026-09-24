"""Public Python API for Loom.

Use the :class:`Loom` client for both in-memory and file-based workflows::

    from loom import Loom

    client = Loom("openai", "gpt-4o-mini", cache_dir="/tmp/loom-cache", temperature=0.2)
    print(client.generate("Say hello"))
    result = client.generate_many(["a", "b", "c"])
    creative = client.generate("Write a haiku", params={"temperature": 1.0, "max_tokens": 60})
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import pandas as pd

from .core import orchestrator
from .core.models import BatchMetadata, GenerationParams, ProviderName, PromptItem, as_params
from .utils import storage
from .utils.cache import ResponseCache

PathLike = Union[str, Path]
ProgressCallback = Callable[[int, int, int, int], None]
ParamsLike = Union[GenerationParams, dict[str, Any], None]


# ---------- Result types ----------

@dataclass
class PromptResult:
    """Outcome for a single prompt."""

    id: str
    prompt: str
    text: str = ""
    error: Optional[str] = None
    cached: bool = False


@dataclass
class GenerationResult:
    """Aggregate result of :meth:`Loom.generate_many`."""

    results: list[PromptResult] = field(default_factory=list)
    total: int = 0
    cache_hits: int = 0
    errors: int = 0

    @property
    def texts(self) -> list[str]:
        """Response texts in input order (empty string on error)."""
        return [r.text for r in self.results]

    @property
    def error_messages(self) -> dict[str, str]:
        return {r.id: r.error for r in self.results if r.error}


@dataclass
class RunResult:
    """Outcome of a file-based sync run (:meth:`Loom.run_file`)."""

    output_path: Path
    total: int
    cache_hits: int
    errors: int
    error_messages: dict[str, str] = field(default_factory=dict)


@dataclass
class TokenCountResult:
    """Outcome of :meth:`Loom.count_tokens`."""

    total_tokens: int
    total: int
    errors: int


@dataclass
class BatchFetchResult:
    """Outcome of :meth:`BatchJob.fetch`."""

    job: "BatchJob"
    done: bool
    responses: dict[str, str] = field(default_factory=dict)
    error_messages: dict[str, str] = field(default_factory=dict)
    output_path: Optional[Path] = None


# ---------- BatchJob ----------

class BatchJob:
    """Handle for a submitted batch job."""

    def __init__(
        self,
        meta: BatchMetadata,
        *,
        api_key: Optional[str] = None,
        loom: Optional["Loom"] = None,
        from_cache: bool = False,
    ) -> None:
        self._meta = meta
        self._api_key = api_key
        self._loom = loom
        self.from_cache = from_cache

    @property
    def id(self) -> str:
        return self._meta.batch_id

    @property
    def status(self) -> str:
        return self._meta.status

    @property
    def meta(self) -> BatchMetadata:
        return self._meta

    @property
    def is_done(self) -> bool:
        return self._meta.status == "completed"

    @property
    def provider(self) -> str:
        return self._meta.provider

    @property
    def model(self) -> str:
        return self._meta.model

    @property
    def output_path(self) -> Optional[Path]:
        return Path(self._meta.output_path) if self._meta.output_path else None

    def refresh(self) -> str:
        """Poll the provider and update ``status``. Returns the new status."""
        if self.from_cache or self._meta.batch_id.startswith("local-"):
            return self._meta.status
        key = self._api_key or (self._loom.api_key if self._loom else None)
        from .providers import get_provider
        from .utils.keys import resolve_api_key

        api_key = resolve_api_key(self._meta.provider, key)
        provider = get_provider(self._meta.provider, api_key)
        status = provider.check_status(self._meta.batch_id)
        self._meta.status = status
        try:
            storage.save_batch(self._meta)
        except Exception:  # noqa: BLE001
            pass
        return status

    def fetch(self, *, force: bool = False, keep: bool = False) -> BatchFetchResult:
        """Download results if the batch is complete.

        For file-sourced jobs the merged output file is written. For in-memory
        jobs the responses are returned in :attr:`BatchFetchResult.responses`.
        Fully-cached local jobs return immediately with the cached responses.
        """
        if self.from_cache or self._meta.batch_id.startswith("local-"):
            return BatchFetchResult(
                job=self,
                done=True,
                responses=dict(self._meta.cached_responses),
                output_path=self.output_path,
            )

        key = self._api_key or (self._loom.api_key if self._loom else None)
        meta, done, errors, responses = orchestrator.fetch_batch(
            self._meta.batch_id,
            api_key=key,
            keep=keep,
            force=force,
        )
        self._meta = meta
        return BatchFetchResult(
            job=self,
            done=done,
            responses=responses,
            error_messages=errors,
            output_path=Path(meta.output_path) if meta.output_path else None,
        )

    def __repr__(self) -> str:
        return (
            f"BatchJob(id={self.id!r}, provider={self.provider!r}, "
            f"model={self.model!r}, status={self.status!r})"
        )


# ---------- Loom client ----------

def _items_from_prompts(prompts: Sequence[str]) -> list[PromptItem]:
    return [
        PromptItem(custom_id=f"prompt-{i}", prompt=str(p))
        for i, p in enumerate(prompts)
    ]


class Loom:
    """Client for running LLM jobs via Loom.

    Parameters
    ----------
    provider:
        ``"openai"``, ``"anthropic"``, ``"google"``, or ``"openrouter"``.
    model:
        Provider-specific model id.
    api_key:
        Override the environment / ``.env`` key for this client.
    workers:
        Concurrent workers for sync generation and token counting.
    use_cache:
        When True (default), read/write the on-disk response cache.
    cache_dir:
        Override the cache location (default: ``$LOOM_CACHE_DIR`` or
        ``~/.loom/cache``).
    with_meta:
        When True, file-based outputs also include ``llm_provider`` /
        ``llm_model`` columns/fields.
    temperature, max_tokens, top_p, top_k, stop, seed, presence_penalty, frequency_penalty, system, json_mode, extra:
        Default generation settings for every request of this client (see
        :class:`~loom.core.models.GenerationParams`). ``None`` keeps the
        provider default. A setting the provider does not support raises
        :class:`~loom.utils.errors.UnsupportedParameterError`.
    params:
        The same settings as a :class:`GenerationParams` or dict; the explicit
        keyword arguments above take precedence.

    Every generation method also takes ``params=`` to override settings for one call.
    Responses are cached per settings: the same prompt with another temperature is a
    separate cache entry.
    """

    def __init__(
        self,
        provider: ProviderName,
        model: str,
        *,
        api_key: Optional[str] = None,
        workers: int = 8,
        use_cache: bool = True,
        cache_dir: Optional[PathLike] = None,
        with_meta: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[Sequence[str]] = None,
        seed: Optional[int] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        system: Optional[str] = None,
        json_mode: Optional[bool] = None,
        extra: Optional[dict[str, Any]] = None,
        params: ParamsLike = None,
    ) -> None:
        self.provider: ProviderName = provider
        self.model = model
        self.api_key = api_key
        self.workers = workers
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self.with_meta = with_meta
        explicit = {
            "temperature": temperature, "max_tokens": max_tokens, "top_p": top_p, "top_k": top_k,
            "stop": list(stop) if stop is not None else None, "seed": seed,
            "presence_penalty": presence_penalty, "frequency_penalty": frequency_penalty,
            "system": system, "json_mode": json_mode, "extra": extra,
        }
        self.params: GenerationParams = as_params(params).merged(
            GenerationParams(**{k: v for k, v in explicit.items() if v is not None})
        )
        self._cache = ResponseCache(
            cache_dir=self.cache_dir,
            enabled=use_cache,
        )

    @property
    def cache(self) -> ResponseCache:
        """The :class:`~loom.utils.cache.ResponseCache` used by this client."""
        return self._cache

    def _params(self, override: ParamsLike) -> GenerationParams:
        """Client settings with ``override``'s explicitly set fields on top."""
        if override is None:
            return self.params
        if isinstance(override, dict):
            override = GenerationParams(**override)
        return self.params.merged(override)

    # ----- in-memory -----

    def generate(self, prompt: str, *, params: ParamsLike = None) -> str:
        """Generate a response for a single prompt. Raises on provider error."""
        result = self.generate_many([prompt], params=params)
        if result.results and result.results[0].error:
            raise RuntimeError(result.results[0].error)
        return result.texts[0] if result.texts else ""

    def generate_many(
        self,
        prompts: Sequence[str],
        *,
        on_progress: Optional[ProgressCallback] = None,
        params: ParamsLike = None,
    ) -> GenerationResult:
        """Generate responses for many prompts concurrently.

        Returns a :class:`GenerationResult` with per-prompt outcomes in input
        order. Individual provider errors are captured on
        :attr:`PromptResult.error` rather than raising.
        """
        items = _items_from_prompts(prompts)
        run = orchestrator.generate_items(
            items,
            self.provider,
            self.model,
            api_key=self.api_key,
            workers=self.workers,
            use_cache=self.use_cache,
            cache=self._cache,
            on_progress=on_progress,
            params=self._params(params),
        )
        results = [
            PromptResult(
                id=r.custom_id,
                prompt=r.prompt,
                text=r.text,
                error=r.error,
                cached=r.cached,
            )
            for r in run.results
        ]
        return GenerationResult(
            results=results,
            total=run.total,
            cache_hits=run.cache_hits,
            errors=run.errors,
        )

    def generate_frame(
        self,
        df: pd.DataFrame,
        column: str = "text",
        *,
        on_progress: Optional[ProgressCallback] = None,
        params: ParamsLike = None,
    ) -> pd.DataFrame:
        """Run every value in ``column`` and return a copy with ``llm_response``.

        When ``with_meta`` is True on the client, also adds ``llm_provider``
        and ``llm_model`` columns.
        """
        if column not in df.columns:
            raise ValueError(
                f"Column '{column}' not found. Available: {list(df.columns)}"
            )
        prompts = [str(v) for v in df[column].tolist()]
        result = self.generate_many(prompts, on_progress=on_progress, params=params)
        out = df.copy()
        out["llm_response"] = result.texts
        if self.with_meta:
            out["llm_provider"] = self.provider
            out["llm_model"] = self.model
        return out

    def count_tokens(
        self,
        prompts: Union[str, Sequence[str]],
        *,
        on_progress: Optional[ProgressCallback] = None,
    ) -> TokenCountResult:
        """Count input tokens for one or more prompts."""
        if isinstance(prompts, str):
            seq: Sequence[str] = [prompts]
        else:
            seq = prompts
        items = _items_from_prompts(seq)
        total_tokens, total, errors = orchestrator.count_token_items(
            items,
            self.provider,
            self.model,
            api_key=self.api_key,
            workers=self.workers,
            on_progress=on_progress,
        )
        return TokenCountResult(total_tokens=total_tokens, total=total, errors=errors)

    # ----- file-based -----

    def run_file(
        self,
        path: PathLike,
        *,
        column: str = "text",
        output: Optional[PathLike] = None,
        force: bool = False,
        on_progress: Optional[ProgressCallback] = None,
        params: ParamsLike = None,
    ) -> RunResult:
        """Synchronously process a JSON/CSV/Parquet file and write the output."""
        out_path, total, hits, errors, error_messages = orchestrator.generate_sync(
            file_path=Path(path),
            provider_name=self.provider,
            model=self.model,
            column=column,
            api_key=self.api_key,
            output_path=Path(output) if output else None,
            workers=self.workers,
            use_cache=self.use_cache,
            force=force,
            with_meta=self.with_meta,
            cache_dir=self.cache_dir,
            on_progress=on_progress,
            params=self._params(params),
        )
        return RunResult(
            output_path=out_path,
            total=total,
            cache_hits=hits,
            errors=errors,
            error_messages=error_messages,
        )

    def submit_file(
        self,
        path: PathLike,
        *,
        column: str = "text",
        output: Optional[PathLike] = None,
        force: bool = False,
        params: ParamsLike = None,
    ) -> BatchJob:
        """Submit a file as a provider batch job. Returns a :class:`BatchJob`."""
        meta = orchestrator.run_batch(
            file_path=Path(path),
            provider_name=self.provider,
            model=self.model,
            column=column,
            api_key=self.api_key,
            output_path=Path(output) if output else None,
            with_meta=self.with_meta,
            use_cache=self.use_cache,
            cache_dir=self.cache_dir,
            force=force,
            params=self._params(params),
        )
        from_cache = meta.batch_id.startswith("local-")
        return BatchJob(meta, api_key=self.api_key, loom=self, from_cache=from_cache)

    def submit(self, prompts: Sequence[str], *, params: ParamsLike = None) -> BatchJob:
        """Submit an in-memory list of prompts as a provider batch job."""
        items = _items_from_prompts(prompts)
        id_map = {it.custom_id: it.custom_id for it in items}

        # Persist a JSON snapshot so fetch can rebuild prompts for the cache.
        # We need a batch_id first for the filename — submit with persist=False
        # after writing a temporary path, then rewrite.
        # Simpler: submit first (which may fabricate a local id), then save input.
        result = orchestrator.submit_items(
            items,
            self.provider,
            self.model,
            api_key=self.api_key,
            use_cache=self.use_cache,
            cache=self._cache,
            id_map=id_map,
            original_file_path="",  # filled in below
            file_type="json",
            with_meta=self.with_meta,
            source="memory",
            persist=False,
            params=self._params(params),
        )

        payload = [{"id": it.custom_id, "prompt": it.prompt} for it in items]
        input_path = storage.save_memory_input(
            self.provider, result.meta.batch_id, payload
        )
        result.meta.original_file_path = str(input_path)

        if result.from_cache:
            return BatchJob(
                result.meta, api_key=self.api_key, loom=self, from_cache=True
            )

        storage.save_batch(result.meta)
        return BatchJob(result.meta, api_key=self.api_key, loom=self)

    # ----- batch lifecycle -----

    def job(self, batch_id: str) -> BatchJob:
        """Load a previously submitted batch by id."""
        meta = storage.load_batch(batch_id)
        return BatchJob(meta, api_key=self.api_key, loom=self)

    def jobs(self) -> list[BatchJob]:
        """List every batch known to Loom."""
        return [
            BatchJob(m, api_key=self.api_key, loom=self)
            for m in storage.list_batches()
        ]


# ---------- Module-level convenience ----------

def generate(
    prompt: str,
    provider: ProviderName,
    model: str,
    *,
    api_key: Optional[str] = None,
    use_cache: bool = True,
    cache_dir: Optional[PathLike] = None,
    params: ParamsLike = None,
    **settings: Any,
) -> str:
    """One-liner: generate a single response. ``settings`` are generation settings, e.g. ``temperature=0.2``."""
    return Loom(
        provider, model, api_key=api_key, use_cache=use_cache, cache_dir=cache_dir, params=params, **settings
    ).generate(prompt)


def generate_many(
    prompts: Sequence[str],
    provider: ProviderName,
    model: str,
    *,
    api_key: Optional[str] = None,
    workers: int = 8,
    use_cache: bool = True,
    cache_dir: Optional[PathLike] = None,
    on_progress: Optional[ProgressCallback] = None,
    params: ParamsLike = None,
    **settings: Any,
) -> GenerationResult:
    """One-liner: generate responses for many prompts. ``settings`` as in :func:`generate`."""
    return Loom(
        provider,
        model,
        api_key=api_key,
        workers=workers,
        use_cache=use_cache,
        cache_dir=cache_dir,
        params=params,
        **settings,
    ).generate_many(prompts, on_progress=on_progress)


def run_file(
    path: PathLike,
    provider: ProviderName,
    model: str,
    *,
    column: str = "text",
    api_key: Optional[str] = None,
    output: Optional[PathLike] = None,
    workers: int = 8,
    use_cache: bool = True,
    cache_dir: Optional[PathLike] = None,
    force: bool = False,
    with_meta: bool = False,
    on_progress: Optional[ProgressCallback] = None,
    params: ParamsLike = None,
    **settings: Any,
) -> RunResult:
    """One-liner: synchronously process a file. ``settings`` as in :func:`generate`."""
    return Loom(
        provider,
        model,
        api_key=api_key,
        workers=workers,
        use_cache=use_cache,
        cache_dir=cache_dir,
        with_meta=with_meta,
        params=params,
        **settings,
    ).run_file(
        path,
        column=column,
        output=output,
        force=force,
        on_progress=on_progress,
    )
