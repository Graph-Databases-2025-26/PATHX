from __future__ import annotations

import logging
import os
from typing import Tuple

from dotenv import load_dotenv
from langchain_ibm import ChatWatsonx
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)

load_dotenv()

# Load Watsonx configuration from environment variables
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID", "ibm/granite-3-3-8b-instruct")

if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
    raise RuntimeError(
        "WATSONX_APIKEY (or WATSONX_API_KEY) and WATSONX_PROJECT_ID must be set in the environment."
    )

_params = {
    "temperature": 0,
}

_llm: ChatWatsonx | None = None


def _get_llm() -> ChatWatsonx:
    """
    Creates and returns the LangChain ChatWatsonx client.
    """
    global _llm
    if _llm is None:
        logger.info("Creating LangChain ChatWatsonx client...")
        _llm = ChatWatsonx(
            model_id=WATSONX_MODEL_ID,
            url=WATSONX_URL,
            project_id=WATSONX_PROJECT_ID,
            apikey=WATSONX_API_KEY,
            params=_params,
        )
    return _llm


def _extract_text(raw_response: BaseMessage) -> str:
    """
    Extracts and returns the generated text from a Watsonx response.
    """
    content = getattr(raw_response, "content", "")
    if isinstance(content, str):
        return content
    return str(content)


def _extract_tokens(raw_response: BaseMessage) -> int | None:
    """
    Extracts and returns the total token count from a Watsonx response, if available.
    """
    usage = getattr(raw_response, "usage_metadata", None)
    if not isinstance(usage, dict):
        logger.warning("Watsonx response does not contain usage_metadata for token count.")
        return None

    in_toks = usage.get("input_tokens")
    out_toks = usage.get("output_tokens")

    if in_toks is None and out_toks is None:
        logger.warning(
            "Watsonx usage_metadata found, but input_tokens/output_tokens are missing or None."
        )
        return None

    total = (in_toks or 0) + (out_toks or 0)
    logger.debug(
        "Watsonx tokens received via LangChain: Total=%d (Input=%s, Output=%s)",
        total,
        in_toks,
        out_toks,
    )
    return total


def generate_text(prompt: str) -> Tuple[str, int | None]:
    """
    Sends a prompt to the Watsonx generative model and returns the generated text and token count
    """
    llm = _get_llm()

    try:
        raw_response = llm.invoke(prompt)
    except Exception as e:
        logger.error("Watsonx LangChain generate_text failed: %s", e)
        raise

    text = _extract_text(raw_response)
    tokens = _extract_tokens(raw_response)

    return text, tokens
