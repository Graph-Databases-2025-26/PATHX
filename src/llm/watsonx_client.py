# src/llm/watsonx_client.py
from __future__ import annotations

import logging
import os
from typing import Tuple
from dotenv import load_dotenv

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials


logger = logging.getLogger(__name__)

load_dotenv()

WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")
WATSONX_PROJECT_ID = os.getenv("WATSONX_PROJECT_ID")
WATSONX_MODEL_ID = os.getenv("WATSONX_MODEL_ID", "ibm/granite-3-3-8b-instruct")

if not WATSONX_API_KEY or not WATSONX_PROJECT_ID:
    raise RuntimeError(
        "WATSONX_API_KEY and WATSONX_PROJECT_ID must be set in the environment."
    )

# --- single persistent model object ---
_creds = Credentials(
    api_key=WATSONX_API_KEY,
    url=WATSONX_URL,
)

_params = {
    "max_new_tokens": 512,
    "temperature": 0.2,
}

_model = ModelInference(
    model_id=WATSONX_MODEL_ID,
    project_id=WATSONX_PROJECT_ID,
    credentials=_creds,
    params=_params,
)


def generate_text(prompt: str) -> Tuple[str, int | None]:
    """
    Unified interface for sql_baseline:
    returns (text, tokens) where tokens may be None if usage isn't available.
    """
    resp = _model.generate_text(prompt=prompt)

    # main generated text
    text = ""
    if isinstance(resp, dict):
        results = resp.get("results") or []
        if results and isinstance(results, list):
            text = results[0].get("generated_text", "") or ""
    else:
        # if not a dict, try to convert to string
        text = str(resp)

    # token amounts (if SDK returns usage)
    tokens = None
    if isinstance(resp, dict):
        usage = resp.get("usage") or {}
        in_toks = usage.get("input_tokens")
        out_toks = usage.get("output_tokens")
        if in_toks is not None or out_toks is not None:
            tokens = (in_toks or 0) + (out_toks or 0)

    return text, tokens
