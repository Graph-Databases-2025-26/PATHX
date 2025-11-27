from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List

from .response_model import PARSER, Response

logger = logging.getLogger(__name__)


def strip_markdown_fences(text: str | None) -> str:
    if text is None:
        return ""
    text = text.strip()

    if text.startswith("00\n\n"):
        text = text[4:]

    if text.startswith("```"):
        first_nl = text.find("\n")
        if first_nl != -1:
            text = text[first_nl + 1 :]
        last_fence = text.rfind("```")
        if last_fence != -1:
            text = text[:last_fence]

    return text.strip()


def _parse_loose_json(text: str) -> List[Dict[str, Any]]:
    # 1) direct
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            return [parsed]
        return [{"_value": parsed}]
    except Exception:
        pass

    # 2) arrays
    for cand in re.findall(r"\[[\s\S]*?\]", text):
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
            return [{"_value": parsed}]
        except Exception:
            continue

    # 3) objects
    for cand in re.findall(r"\{[\s\S]*?\}", text):
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                return [parsed]
            return [{"_value": parsed}]
        except Exception:
            continue

    logger.warning("Failed to parse LLM output as JSON, storing raw text.")
    return [{"_error": "failed_to_parse_json", "raw_text": text}]


def parse_llm_response(
    raw_answer: Any,
    elapsed: float | None,
    tokens: Any,
) -> Dict[str, Any]:
    """
    Unified parsing for both NL and SQL baselines.
    """
    text = strip_markdown_fences(str(raw_answer))

    try:
        parsed: Response = PARSER.parse(text)
        result_set = parsed.result_set
    except Exception as e:
        logger.warning("Pydantic parsing failed (%s), falling back to loose JSON.", e)
        result_set = _parse_loose_json(text)

    return {
        "result_set": result_set,
        "time": round(elapsed, 3) if elapsed is not None else None,
        "tokens": tokens,
    }
