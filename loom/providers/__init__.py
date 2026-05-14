from .base import BatchProvider, BatchStatus, BatchResult
from .openai import OpenAIProvider
from .anthropic import AnthropicProvider
from .google import GoogleProvider


def get_provider(name: str, api_key: str | None = None) -> BatchProvider:
    name = name.lower()
    if name in ("openai", "oai"):
        return OpenAIProvider(api_key=api_key)
    if name in ("anthropic", "claude"):
        return AnthropicProvider(api_key=api_key)
    if name in ("google", "gemini"):
        return GoogleProvider(api_key=api_key)
    raise ValueError(f"Unknown provider: {name}. Use openai, anthropic, or google.")
