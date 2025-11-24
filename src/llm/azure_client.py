from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from openai import AzureOpenAI, APIError

logger = logging.getLogger(__name__)

load_dotenv()

def _build_client() -> AzureOpenAI:
    """ 
    Builds and returns an Azure OpenAI client using environment variables. 
    """
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is not set.")
    if not api_key:
        raise RuntimeError("AZURE_OPENAI_API_KEY is not set.")

    logger.info(
        f"Initializing Azure OpenAI client "
        f"(endpoint={endpoint}, api_version={api_version})"
    )

    return AzureOpenAI(
        azure_endpoint=endpoint,
        api_key=api_key,
        api_version=api_version,
    )


_client = _build_client()

_AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")


def generate_text(
    prompt: str,
    *,
    temperature: float = 0.0,
    max_output_tokens: int = 2048,
) -> tuple[str, int | None]:
    """
    Simple wrapper: send a prompt to Azure OpenAI chat model
    and return (text, total_tokens).
    """

    try:
        # Azure does not support the Responses API; it uses chat completions instead.
        response = _client.chat.completions.create(
            model=_AZURE_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_output_tokens,
        )
    except APIError as e:
        logger.error(f"Azure OpenAI APIError: {e}")
        raise
    except Exception as e:
        logger.error(f"Azure OpenAI unexpected error: {e}")
        raise

    tokens: int | None = None
    try:
        usage = getattr(response, "usage", None)
        if usage is not None:
            if hasattr(usage, "total_tokens"):
                tokens = getattr(usage, "total_tokens")
            elif isinstance(usage, dict):
                tokens = usage.get("total_tokens")

        # Standard chat completion shape
        content = response.choices[0].message.content
        if isinstance(content, list):
            # Sometimes Azure returns a list of content parts
            text_parts = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
                elif isinstance(part, str):
                    text_parts.append(part)
            text = "".join(text_parts)
        else:
            text = content or ""
    except Exception as e:
        logger.error(f"Failed to extract text from Azure response: {e}")
        raise RuntimeError(
            f"Azure response has unexpected format: {response!r}"
        )

    return text, tokens
