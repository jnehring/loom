"""API key resolution.

Precedence: explicit flag > environment variable > .env file.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import find_dotenv, load_dotenv

from ..core.models import ProviderName

# Load .env once at import-time (does not override existing env vars). Search from the current
# working directory: without usecwd, python-dotenv searches upward from this file's location,
# i.e. the installed package, and misses the .env of the project that uses Loom.
load_dotenv(find_dotenv(usecwd=True), override=False)


ENV_VAR = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def resolve_api_key(provider: ProviderName, explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    env_name = ENV_VAR[provider]
    key = os.environ.get(env_name)
    if not key:
        raise RuntimeError(
            f"No API key for provider '{provider}'. "
            f"Set ${env_name} in your environment or .env, or pass --api-key."
        )
    return key
