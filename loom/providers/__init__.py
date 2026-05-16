"""Provider strategies for batch LLM submission and synchronous generation."""

from .base import BatchProvider
from .sync_base import SyncProvider
from ..core.models import ProviderName


def get_provider(name: ProviderName, api_key: str) -> BatchProvider:
    if name == "openai":
        from .openai import OpenAIBatchProvider
        return OpenAIBatchProvider(api_key=api_key)
    if name == "anthropic":
        from .anthropic import AnthropicBatchProvider
        return AnthropicBatchProvider(api_key=api_key)
    if name == "google":
        from .google import GoogleBatchProvider
        return GoogleBatchProvider(api_key=api_key)
    if name == "openrouter":
        raise ValueError(
            "OpenRouter has no batch API. Use 'loom run --sync --provider openrouter ...' instead."
        )
    raise ValueError(f"Unknown provider: {name}")


def get_sync_provider(name: ProviderName, api_key: str) -> SyncProvider:
    if name == "openai":
        from .openai_sync import OpenAISyncProvider
        return OpenAISyncProvider(api_key=api_key)
    if name == "anthropic":
        from .anthropic_sync import AnthropicSyncProvider
        return AnthropicSyncProvider(api_key=api_key)
    if name == "google":
        from .google_sync import GoogleSyncProvider
        return GoogleSyncProvider(api_key=api_key)
    if name == "openrouter":
        from .openrouter_sync import OpenRouterSyncProvider
        return OpenRouterSyncProvider(api_key=api_key)
    raise ValueError(f"Unknown provider: {name}")
