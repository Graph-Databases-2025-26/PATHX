from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Set, Tuple

from .parsing import parse_llm_response

logger = logging.getLogger(__name__)

def extract_tokens(obj: Any) -> int | None:
    """
    Universal token extractor from LLM response objects.
    """
    if obj is None:
        return None

    # OpenAI / Azure
    usage = getattr(obj, "usage", None)
    if usage is not None:
        for field in ["total_tokens", "completion_tokens", "output_tokens"]:
            if hasattr(usage, field):
                return getattr(usage, field)

        if isinstance(usage, dict):
            for field in ["total_tokens", "completion_tokens", "output_tokens"]:
                if field in usage:
                    return usage[field]

    # Gemini / WatsonX
    usage2 = getattr(obj, "usage_metadata", None)
    if usage2 is not None:
        for field in ["total_tokens", "output_tokens", "input_tokens"]:
            if hasattr(usage2, field):
                return getattr(usage2, field)

        if isinstance(usage2, dict):
            for field in ["total_tokens", "output_tokens"]:
                if field in usage2:
                    return usage2[field]

    return None
 
def _tuples_signature(rows: List[Dict[str, Any]]) -> List[Tuple]:
    """
    Make each dict tuple(sorted(items)) for the comparison of tuples.
    """
    sigs: List[Tuple] = []
    for row in rows:
        if isinstance(row, dict):
            sigs.append(tuple(sorted(row.items())))
        else:
            sigs.append((("__raw__", str(row)),))
    return sigs


def run_iterative_baseline(
    initial_chain,
    followup_chain,
    initial_payload: Dict[str, Any],
    max_iters: int = 5,
):
    """
    initial_chain.invoke(payload) makes first batch
    followup_chain.invoke({}) makes follow-up iterations "more"
    """
    all_rows: List[Dict[str, Any]] = []
    seen: Set[Tuple] = set()
    total_time = 0.0
    total_tokens = 0
    used_iters = 0

    # First iteration
    t0 = time.time()
    raw_first = initial_chain.invoke(initial_payload)
    elapsed_first = time.time() - t0
    total_time += elapsed_first

    out_tokens = extract_tokens(raw_first)
    if out_tokens:
        total_tokens += out_tokens

    parsed_first = parse_llm_response(
        raw_answer=getattr(raw_first, "content", raw_first),
        elapsed=elapsed_first,
        tokens=out_tokens,
    )

    rows = parsed_first["result_set"]
    sigs = _tuples_signature(rows)

    new_rows = []
    for row, sig in zip(rows, sigs):
        if sig not in seen:
            seen.add(sig)
            new_rows.append(row)

    all_rows.extend(new_rows)
    used_iters += 1

    logger.info(
        f"[ITER] iter=0 (initial): rows={len(rows)}, new={len(new_rows)}, "
        f"time={elapsed_first:.3f}s, tokens={out_tokens}"
    )

    # Follow-up iterations
    for i in range(1, max_iters + 1):
        t0 = time.time()
        raw_next = followup_chain.invoke({})
        elapsed_next = time.time() - t0
        total_time += elapsed_next

        next_tokens = extract_tokens(raw_next)
        if next_tokens:
            total_tokens += next_tokens

        parsed_next = parse_llm_response(
            raw_answer=getattr(raw_next, "content", raw_next),
            elapsed=elapsed_next,
            tokens=next_tokens,
        )

        rows_next = parsed_next["result_set"]
        sigs_next = _tuples_signature(rows_next)

        only_new = []
        for row, sig in zip(rows_next, sigs_next):
            if sig not in seen:
                seen.add(sig)
                only_new.append(row)

        logger.info(
            f"[ITER] iter={i}: rows={len(rows_next)}, new={len(only_new)}, "
            f"time={elapsed_next:.3f}s, tokens={next_tokens}"
        )

        if not only_new:
            logger.info(f"[ITER] stopping at iter={i}: no new rows")
            break

        all_rows.extend(only_new)
        used_iters += 1

    return {
        "result_set": all_rows,
        "time": round(total_time, 3),
        "tokens": total_tokens,       # sum of output tokens
        "meta": {
            "iterations_used": used_iters,
        },
    }