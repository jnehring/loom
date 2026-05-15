"""Provider strategies for batch LLM submission."""

from .base import BatchProvider
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
    raise ValueError(f"Unknown provider: {name}")
