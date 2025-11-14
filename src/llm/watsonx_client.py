from __future__ import annotations
from pathlib import Path
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# Load .env once (repo root)
load_dotenv(dotenv_path=Path(__file__).resolve().parents[2] / ".env")

def _env(name: str, default: Optional[str] = None) -> str:
    v = os.getenv(name, default)
    if v is None:
        raise RuntimeError(f"Missing required env var: {name}")
    return v

def build_model(
    model_id: Optional[str] = None,
    temperature: Optional[float] = None,
    max_new_tokens: Optional[int] = None,
):
    """Create a watsonx.ai text model instance (Foundation Models API)."""
    api_key   = _env("WATSONX_API_KEY")
    url       = _env("WATSONX_URL")
    project_id = os.getenv("WATSONX_PROJECT_ID")
    space_id   = os.getenv("WATSONX_SPACE_ID") 

    model_id   = model_id or os.getenv("WATSONX_MODEL_ID", "ibm/granite-20b-multilingual")
    temperature = float(temperature if temperature is not None else os.getenv("WATSONX_TEMPERATURE", 0.2))
    max_new_tokens = int(max_new_tokens if max_new_tokens is not None else os.getenv("WATSONX_MAX_NEW_TOKENS", 256))

    creds = Credentials(api_key=api_key, url=url)

    params = {
        GenParams.TEMPERATURE: temperature,
        GenParams.MAX_NEW_TOKENS: max_new_tokens,
        # optional:
        # GenParams.TOP_P: 0.9,
        # GenParams.REPETITION_PENALTY: 1.05,
        # GenParams.STOP_SEQUENCES: ["```"]
    }

    kwargs: Dict[str, Any] = dict(model_id=model_id, params=params, credentials=creds)
    if project_id:
        kwargs["project_id"] = project_id
    elif space_id:
        kwargs["space_id"] = space_id
    else:
        raise RuntimeError("Provide either WATSONX_PROJECT_ID or WATSONX_SPACE_ID")

    return Model(**kwargs)

def generate_text(prompt: str, **overrides) -> str:
    """Simple text generation wrapper; returns plain string."""
    model = build_model(**overrides)
    # Most SDK versions expose .generate() returning text; some return dict.
    out = model.generate(prompt)
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        # try common locations
        for key in ("results", "generated_text", "text"):
            if key in out:
                val = out[key]
                if isinstance(val, list) and val and isinstance(val[0], dict) and "generated_text" in val[0]:
                    return val[0]["generated_text"]
                if isinstance(val, str):
                    return val
        return str(out)
    # fallback
    return str(out)
