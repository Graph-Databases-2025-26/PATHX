from __future__ import annotations

import os
from abc import ABC, abstractmethod

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ibm import ChatWatsonx
from langchain_openai import AzureChatOpenAI, ChatOpenAI 

from src.utils.logging import logger as LOG

class LLMBaseWrapper(ABC):
    def __init__(self, config):
        self.config = config
        self.llm_instance: BaseChatModel | None = None

    @abstractmethod
    def _create_llm_instance(self, config) -> BaseChatModel: ...

    @abstractmethod
    def get_output_tokens(self, raw_response: BaseMessage) -> int: ...

    @abstractmethod
    def get_provider_name(self) -> str: ...

    def get_llm_instance(self) -> BaseChatModel:
        if not self.llm_instance:
            self.llm_instance = self._create_llm_instance(self.config)
        return self.llm_instance


class WatsonxWrapper(LLMBaseWrapper):
    def get_provider_name(self) -> str:
        return "watsonx"

    def _create_llm_instance(self, config):
        return ChatWatsonx(
            model_id=config.watsonx.model,
            api_key=config.watsonx.api_key,
            url=config.watsonx.endpoint,
            project_id=config.watsonx.project_id,
            params={
                "temperature": config.watsonx.temperature,
                "max_new_tokens": config.watsonx.max_tokens,
            }
        )

    def get_output_tokens(self, raw_response: BaseMessage) -> int:
        return raw_response.usage_metadata.get("output_tokens", 0)


class GeminiWrapper(LLMBaseWrapper):
    def get_provider_name(self) -> str:
        return "gemini"

    def _create_llm_instance(self, config):
        return ChatGoogleGenerativeAI(
            model=config.gemini.model,
            temperature=config.gemini.temperature,
            max_output_tokens=config.gemini.max_output_tokens,
            google_api_key=config.gemini.api_key,
        )

    def get_output_tokens(self, raw_response: BaseMessage) -> int:
        return raw_response.usage_metadata.get("output_tokens", 0)


class AzureOpenAIWrapper(LLMBaseWrapper):
    def get_provider_name(self) -> str:
        return "azure"

    def _create_llm_instance(self, config) -> BaseChatModel:
        endpoint    = getattr(config.azure, "endpoint", None)    or os.getenv("AZURE_OPENAI_ENDPOINT")
        api_key     = getattr(config.azure, "api_key", None)     or os.getenv("AZURE_OPENAI_API_KEY")
        api_version = getattr(config.azure, "api_version", None) or os.getenv("AZURE_OPENAI_API_VERSION")
        deployment  = getattr(config.azure, "deployment", None)  or os.getenv("AZURE_OPENAI_DEPLOYMENT")

        if not (endpoint and api_key and api_version and deployment):
            raise RuntimeError("Azure OpenAI config is incomplete (endpoint/api_key/api_version/deployment).")

        LOG.info(f"Creating Azure OpenAI LLM (deployment={deployment}, api_version={api_version})")

        return AzureChatOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
            deployment_name=deployment,
            temperature=config.azure.temperature,
            max_tokens=config.azure.max_tokens,
        )

    def get_output_tokens(self, raw_response: BaseMessage) -> int:
        usage = getattr(raw_response, "usage_metadata", None) or getattr(raw_response, "usage", None)
        if isinstance(usage, dict):
            return usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
        return 0
    
class AzureLlamaWrapper(LLMBaseWrapper):
    def get_provider_name(self) -> str:
        return "azure_llama"

    def _create_llm_instance(self, config) -> BaseChatModel:
        endpoint = getattr(config.azure_llama, "endpoint", None) or os.getenv("AZURE_LLAMA_ENDPOINT")
        api_key  = getattr(config.azure_llama, "api_key", None)  or os.getenv("AZURE_LLAMA_API_KEY")
        model    = getattr(config.azure_llama, "deployment", None) or os.getenv("AZURE_LLAMA_DEPLOYMENT")
        temperature = config.azure_llama.temperature
        max_tokens  = config.azure_llama.max_tokens

        if not (endpoint and api_key and model):
            raise RuntimeError(
                "Azure Llama config is incomplete "
                "(endpoint/api_key/deployment). Check AppConfig.yaml and .env."
            )

        LOG.info(f"Creating Azure Llama LLM (deployment={model})")

        return ChatOpenAI(
            base_url=endpoint,
            api_key=api_key,
            model=model,        # "Llama-3.3-70B-Instruct"
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def get_output_tokens(self, raw_response: BaseMessage) -> int:
        usage = getattr(raw_response, "usage_metadata", None) or getattr(raw_response, "usage", None)
        if isinstance(usage, dict):
            return usage.get("completion_tokens", 0) or usage.get("output_tokens", 0)
        return 0
