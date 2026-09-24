"""Alibaba Cloud Model Studio (DashScope, Qwen models): batch and sync over the OpenAI-compatible API.

Docs: https://www.alibabacloud.com/help/en/model-studio/batch-interfaces-compatible-with-openai/

- API key: ``$DASHSCOPE_API_KEY``. Keys, endpoints and model lists are per region and cannot be mixed.
- Endpoint: ``$DASHSCOPE_BASE_URL``, default Singapore (``https://dashscope-intl.aliyuncs.com/compatible-mode/v1``).
  Beijing: ``https://dashscope.aliyuncs.com/compatible-mode/v1``; workspace endpoints look like
  ``https://{WorkspaceId}.{region}.maas.aliyuncs.com/compatible-mode/v1`` (e.g. ``eu-central-1``, ``us-east-1``).
- Batch: same file format and flow as OpenAI (``/v1/chat/completions``, ``24h`` window), half the real-time price. One
  model and one thinking mode per batch file; batch models per region are listed in the docs (Singapore: qwen-max,
  qwen-plus, qwen-flash, qwen-turbo).
- Model-specific switches such as ``enable_thinking`` go into ``GenerationParams.extra`` (sent as ``extra_body``).
"""

from __future__ import annotations

import os
from typing import Optional

from .openai import OpenAIBatchProvider
from .openai_sync import OpenAISyncProvider

ALIBABA_DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"


def alibaba_base_url() -> str:
    return os.environ.get("DASHSCOPE_BASE_URL") or ALIBABA_DEFAULT_BASE_URL


class AlibabaBatchProvider(OpenAIBatchProvider):
    name = "alibaba"

    def __init__(self, api_key: str) -> None:
        self.base_url = alibaba_base_url()
        super().__init__(api_key)


class AlibabaSyncProvider(OpenAISyncProvider):
    name = "alibaba"

    def __init__(self, api_key: str) -> None:
        self.base_url = alibaba_base_url()
        super().__init__(api_key)

    def count_tokens(self, prompt: str, model: str) -> Optional[int]:
        return None  # no remote token-counting API
