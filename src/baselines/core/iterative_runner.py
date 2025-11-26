from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Set, Tuple

from .parsing import parse_llm_response

logger = logging.getLogger(__name__)


def _tuples_signature(rows: List[Dict[str, Any]]) -> Set[Tuple]:
    """
    Signature for each row to detect duplicates.
    """
    sigs: Set[Tuple] = set()
    for row in rows:
        if isinstance(row, dict):
            sigs.add(tuple(sorted(row.items())))
    return sigs


def _extract_total_tokens(raw_response) -> int:
    """
    Extract total tokens (input + output) for a single LLM call.
    Works for watsonx, Azure (usage) and Gemini (usage_metadata).
    Falls back to 0 if no metadata is available.
    """
    # Preferred: usage_metadata (watsonx, gemini)
    usage_meta = getattr(raw_response, "usage_metadata", None)
    if usage_meta:
        inp = usage_meta.get("input_tokens", 0)
        out = usage_meta.get("output_tokens", 0)
        return int(inp) + int(out)

    # Azure / OpenAI-style: raw_response.usage
    usage = getattr(raw_response, "usage", None)
    if usage is not None:
        # Some SDKs expose as attributes, some as dict
        if hasattr(usage, "prompt_tokens") or hasattr(usage, "completion_tokens"):
            inp = getattr(usage, "prompt_tokens", 0)
            out = getattr(usage, "completion_tokens", 0)
        elif isinstance(usage, dict):
            inp = usage.get("prompt_tokens", 0)
            out = usage.get("completion_tokens", 0)
        else:
            inp = out = 0
        return int(inp) + int(out)

    return 0


def run_iterative_baseline(
    initial_chain,
    followup_chain,
    initial_payload: Dict[str, Any],
    max_iters: int = 5,
) -> Dict[str, Any]:
    """
    Perform:
      - initial LLM call (iter=0)
      - up to max_iters follow-up calls (iter=1..N)

    Aggregates:
      - all unique rows across iterations
      - total wall-clock time (sum of per-call durations)
      - total tokens (sum of input+output tokens for all calls)

    Returns:
      {
        "result_set": [...],
        "time": <total_time_seconds>,
        "tokens": <total_tokens>,
        "meta": { "iterations_used": <num_successful_followups> }
      }
    """

    all_rows: List[Dict[str, Any]] = []
    seen: Set[Tuple] = set()

    total_tokens = 0
    total_time = 0.0
    iterations_used = 0  # number of follow-up iterations that yielded new rows

    # initial call (iter=0)
    t0 = time.time()
    raw_first = initial_chain.invoke(initial_payload)
    elapsed = time.time() - t0

    iter_tokens = _extract_total_tokens(raw_first)
    total_time += elapsed
    total_tokens += iter_tokens

    first_parsed = parse_llm_response(
        raw_answer=raw_first.content if hasattr(raw_first, "content") else raw_first,
        elapsed=elapsed,
        tokens=None,
    )
    rows = first_parsed["result_set"]

    # deduplicate initial rows
    sigs = _tuples_signature(rows)
    new_rows = []
    for row, sig in zip(rows, sigs):
        if sig not in seen:
            seen.add(sig)
            new_rows.append(row)

    all_rows.extend(new_rows)

    logger.info(
        "[ITER] iter=0 (initial): rows=%d, new=%d, time=%.3fs, tokens=%d",
        len(rows),
        len(new_rows),
        elapsed,
        iter_tokens,
    )

    # follow-up iterations
    for i in range(1, max_iters + 1):
        t0 = time.time()
        raw_next = followup_chain.invoke({})
        elapsed = time.time() - t0

        iter_tokens = _extract_total_tokens(raw_next)
        total_time += elapsed
        total_tokens += iter_tokens

        parsed_next = parse_llm_response(
            raw_answer=raw_next.content if hasattr(raw_next, "content") else raw_next,
            elapsed=elapsed,
            tokens=None,
        )
        rows_next = parsed_next["result_set"]

        sigs_next = _tuples_signature(rows_next)
        only_new: List[Dict[str, Any]] = []
        for row, sig in zip(rows_next, sigs_next):
            if sig not in seen:
                seen.add(sig)
                only_new.append(row)

        logger.info(
            "[ITER] iter=%d: rows=%d, new=%d, time=%.3fs, tokens=%d",
            i,
            len(rows_next),
            len(only_new),
            elapsed,
            iter_tokens,
        )

        if not only_new:
            logger.info("[ITER] stopping at iter=%d: no new rows", i)
            break

        iterations_used += 1
        all_rows.extend(only_new)

    return {
        "result_set": all_rows,
        "time": round(total_time, 3),
        "tokens": int(total_tokens),
        "meta": {"iterations_used": iterations_used},
    }
