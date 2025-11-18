from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv
import google.generativeai as genai

logger = logging.getLogger(__name__)

load_dotenv()


def _build_model() -> genai.GenerativeModel:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set.")

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    generation_config = {
        "temperature": float(os.getenv("GEMINI_TEMPERATURE", 0.0)),
        "top_p": float(os.getenv("GEMINI_TOP_P", 1.0)),
        "top_k": int(os.getenv("GEMINI_TOP_K", 32)),
        "max_output_tokens": int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", 16384)),
    }

    logger.info(f"Initializing Gemini client (model={model_name})")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name,
        generation_config=generation_config,
    )


_model = _build_model()


def _extract_text(response: Any) -> str:

    if not getattr(response, "candidates", None):
        raise RuntimeError(f"Gemini returned no candidates: {response!r}")

    cand = response.candidates[0]

    finish_reason = getattr(cand, "finish_reason", None)
    if finish_reason not in (None, 0):  # 0 = FINISH_REASON_UNSPECIFIED / OK
        logger.warning("Gemini finish_reason=%r", finish_reason)

    content = getattr(cand, "content", None)
    parts = getattr(content, "parts", None) if content is not None else None
    if not parts:
        raise RuntimeError(f"Gemini candidate has no parts (finish_reason={finish_reason!r})")

    chunks: list[str] = []
    for p in parts:
        t = getattr(p, "text", None)
        if t:
            chunks.append(t)

    if not chunks:
        raise RuntimeError(f"Gemini parts contain no text (finish_reason={finish_reason!r})")

    return "".join(chunks)


def generate_text(prompt: str) -> tuple[str, int | None]:
    try:
        response = _model.generate_content(
            prompt,
            request_options={"timeout": 60},
        )
    except Exception as e:
        logger.error("Gemini generate_content failed: %s", e)
        raise

    tokens: int | None = None
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        # usage_metadata often exposes total_token_count; fall back to None
        tokens = getattr(usage, "total_token_count", None)

    return _extract_text(response), tokens
