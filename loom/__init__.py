"""Loom — weave batch LLM jobs across providers.

Public API (lazy-loaded via ``__getattr__``)::

    from loom import Loom, GenerationParams, generate, generate_many, run_file
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

__version__ = "0.6.0"

__all__ = [
    "Loom",
    "BatchJob",
    "BatchFetchResult",
    "GenerationResult",
    "PromptResult",
    "RunResult",
    "TokenCountResult",
    "ResponseCache",
    "BatchMetadata",
    "PromptItem",
    "GenerationParams",
    "UnsupportedParameterError",
    "OutputExistsError",
    "SyncOutputExistsError",
    "TokenCountingNotSupported",
    "generate",
    "generate_many",
    "run_file",
    "__version__",
]

if TYPE_CHECKING:  # pragma: no cover
    from .api import (
        BatchFetchResult,
        BatchJob,
        GenerationResult,
        Loom,
        PromptResult,
        RunResult,
        TokenCountResult,
        generate,
        generate_many,
        run_file,
    )
    from .core.models import BatchMetadata, GenerationParams, PromptItem
    from .utils.errors import UnsupportedParameterError
    from .core.orchestrator import (
        OutputExistsError,
        SyncOutputExistsError,
        TokenCountingNotSupported,
    )
    from .utils.cache import ResponseCache


_LAZY_MAP = {
    "Loom": (".api", "Loom"),
    "BatchJob": (".api", "BatchJob"),
    "BatchFetchResult": (".api", "BatchFetchResult"),
    "GenerationResult": (".api", "GenerationResult"),
    "PromptResult": (".api", "PromptResult"),
    "RunResult": (".api", "RunResult"),
    "TokenCountResult": (".api", "TokenCountResult"),
    "generate": (".api", "generate"),
    "generate_many": (".api", "generate_many"),
    "run_file": (".api", "run_file"),
    "ResponseCache": (".utils.cache", "ResponseCache"),
    "BatchMetadata": (".core.models", "BatchMetadata"),
    "PromptItem": (".core.models", "PromptItem"),
    "GenerationParams": (".core.models", "GenerationParams"),
    "UnsupportedParameterError": (".utils.errors", "UnsupportedParameterError"),
    "OutputExistsError": (".core.orchestrator", "OutputExistsError"),
    "SyncOutputExistsError": (".core.orchestrator", "SyncOutputExistsError"),
    "TokenCountingNotSupported": (".core.orchestrator", "TokenCountingNotSupported"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_MAP:
        import importlib

        module_name, attr = _LAZY_MAP[name]
        module = importlib.import_module(module_name, __name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
