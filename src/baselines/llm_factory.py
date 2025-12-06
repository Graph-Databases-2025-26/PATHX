from __future__ import annotations
from typing import Any

from .llm_provider import (
    LLMBaseWrapper,
    WatsonxWrapper,
    GeminiWrapper,
    AzureOpenAIWrapper,
    AzureLlamaWrapper,
)


PROVIDER_MAP = {
    "watsonx": WatsonxWrapper,
    "gemini": GeminiWrapper,
    "azure": AzureOpenAIWrapper,
    "azure_llama": AzureLlamaWrapper,
}


def get_llm_wrapper(config: Any) -> LLMBaseWrapper:
    provider = config.llm_provider.lower()
    if provider not in PROVIDER_MAP:
        raise ValueError(f"Unsupported LLM provider: {provider}")
    return PROVIDER_MAP[provider](config)
