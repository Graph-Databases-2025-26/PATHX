from __future__ import annotations

import time
from typing import Any, Dict, List, Set, Tuple, Optional

from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
)

from src.baselines.core.parsing import parse_llm_response
from src.galois.prompt_builder_llmscan import (
    build_llmscan_first,
    build_llmscan_followup,
)


Row = Dict[str, Any]
Table = List[Row]


def _make_hashable(value: Any) -> Any:
    """
    Convert any JSON-like value into a hashable form.
    Used to build stable signatures for rows.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_make_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple((k, _make_hashable(v)) for k, v in sorted(value.items()))
    return repr(value)


def _row_signature(row: Row) -> Tuple:
    """
    Stable, hashable signature for a single row.
    Order of keys and nested structures does not matter.
    """
    items = [(k, _make_hashable(v)) for k, v in row.items()]
    return tuple(sorted(items))


def _filter_new_rows(rows: Table, seen: Set[Tuple]) -> Table:
    """
    Keep only rows that are not in 'seen', update 'seen' in-place.
    """
    new_rows: Table = []
    for row in rows:
        if not isinstance(row, dict):
            # silently drop malformed rows
            continue
        sig = _row_signature(row)
        if sig in seen:
            continue
        seen.add(sig)
        new_rows.append(row)
    return new_rows


def _extract_total_tokens(raw_response) -> int:
    """
    Extract total tokens (input + output) for a single LLM call.
    Works for watsonx, Azure/OpenAI-style usage and Gemini (usage_metadata).
    Falls back to 0 if no metadata is available.
    """
    usage_meta = getattr(raw_response, "usage_metadata", None)
    if usage_meta:
        inp = usage_meta.get("input_tokens", 0) or 0
        out = usage_meta.get("output_tokens", 0) or 0
        return int(inp) + int(out)

    usage = getattr(raw_response, "usage", None)
    if usage is not None:
        if hasattr(usage, "prompt_tokens") or hasattr(usage, "completion_tokens"):
            inp = getattr(usage, "prompt_tokens", 0) or 0
            out = getattr(usage, "completion_tokens", 0) or 0
        elif isinstance(usage, dict):
            inp = usage.get("prompt_tokens", 0) or 0
            out = usage.get("completion_tokens", 0) or 0
        else:
            inp = out = 0
        return int(inp) + int(out)

    return 0


class LLMScanOperator:
    """
    Base LLMScan operator (Table-Scan implementation, Algorithm 1 in GALOIS).

    Responsibilities:
      - Build first & follow-up prompts via Jinja2 templates.
      - Call the underlying LLM through a simple LCEL chain.
      - Parse JSON responses into a unified result_set.
      - Iteratively retrieve additional tuples (re-iteration).
      - Deduplicate rows across iterations.
      - Track total time, total tokens, and iterations_used.
    """

    def __init__(self, llm, max_iters: int = 5):
        """
        Args:
            llm: LangChain ChatModel / Runnable (wrapper.get_llm_instance()).
            max_iters: maximum number of follow-up iterations.
        """
        self.llm = llm
        self.max_iters = max_iters

        # Generic chat prompt: single human message with injected "{prompt}"
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                HumanMessagePromptTemplate(
                    prompt=PromptTemplate(
                        input_variables=["prompt"],
                        template="{prompt}",
                    )
                )
            ]
        )
        # LCEL chain: {"prompt": str} -> LLM response
        self._chain = chat_prompt | llm

    def _call_llm(self, prompt: str):
        """
        Single LLM call with timing and token extraction.
        """
        t0 = time.time()
        raw = self._chain.invoke({"prompt": prompt})
        elapsed = time.time() - t0
        tokens = _extract_total_tokens(raw)
        return raw, elapsed, tokens

    def table_scan(
        self,
        dataset: str,
        table: str,
        attributes: List[str],
        where_clause: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run a full Table-Scan on (dataset, table, attributes, where_clause).

        This corresponds to the TABLE-SCAN physical operator in GALOIS:
          - First query: ask the LLM for rows.
          - Re-iteration: ask for additional rows, avoiding duplicates.
          - Stop when no new rows are produced or max_iters is reached.
        """
        all_rows: Table = []
        seen: Set[Tuple] = set()

        total_time: float = 0.0
        total_tokens: int = 0
        iterations_used: int = 0

        # initial call
        first_prompt = build_llmscan_first(
            dataset=dataset,
            table=table,
            attributes=attributes,
            where_clause=where_clause,
        )
        raw_first, elapsed, iter_tokens = self._call_llm(first_prompt)
        total_time += float(elapsed or 0.0)
        total_tokens += int(iter_tokens or 0)

        parsed_first = parse_llm_response(
            raw_answer=raw_first.content if hasattr(raw_first, "content") else raw_first,
            elapsed=elapsed,
            tokens=None,
        )
        rows = parsed_first["result_set"]
        new_rows = _filter_new_rows(rows, seen)
        all_rows.extend(new_rows)

        # follow-up iterations
        for i in range(1, self.max_iters + 1):
            if not all_rows:
                # if first iteration already returned nothing, no point in continuing
                break

            followup_prompt = build_llmscan_followup(
                dataset=dataset,
                table=table,
                attributes=attributes,
                previous_rows=all_rows,
                where_clause=where_clause,
            )
            raw_next, elapsed, iter_tokens = self._call_llm(followup_prompt)
            total_time += float(elapsed or 0.0)
            total_tokens += int(iter_tokens or 0)

            parsed_next = parse_llm_response(
                raw_answer=raw_next.content
                if hasattr(raw_next, "content")
                else raw_next,
                elapsed=elapsed,
                tokens=None,
            )
            rows_next = parsed_next["result_set"]
            only_new = _filter_new_rows(rows_next, seen)

            if not only_new:
                # stop condition: no new rows
                break

            iterations_used += 1
            all_rows.extend(only_new)

        return {
            "result_set": all_rows,
            "time": round(float(total_time or 0.0), 3),
            "tokens": int(total_tokens or 0),
            "meta": {"iterations_used": iterations_used},
        }
