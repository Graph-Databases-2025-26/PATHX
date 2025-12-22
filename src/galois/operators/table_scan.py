"""
Table-Scan Operator - Algorithm 1 from the paper
Retrieves complete tuples iteratively
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from .base_scan import BaseScanOperator, _filter_new_rows, _row_signature
from ..prompts.templates import PromptTemplates
from ..utils.response_parser import ResponseParser


class TableScanOperator(BaseScanOperator):
    """
    Table-Scan implementation (Algorithm 1 in GALOIS paper).
    
    Strategy:
    1. Ask LLM for complete tuples in first iteration
    2. Re-iterate to get more tuples until no new ones appear
    3. Deduplicate across iterations
    
    This is more efficient than Key-Scan in terms of LLM calls,
    but may produce lower quality for complex queries.
    """

    def scan(
        self,
        dataset: str,
        table: str,
        attributes: List[str],
        where_clause: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute Table-Scan with iterative retrieval (Algorithm 1 in GALOIS).
        Now follow-up prompts include context of previously retrieved rows.
        """
        all_rows: List[Dict] = []
        seen: Set[Tuple] = set()

        total_time = 0.0
        total_tokens = 0
        iterations_used = 0

        # --- FIRST ITERATION ---
        first_prompt = PromptTemplates.table_scan_first(
            dataset=dataset,
            table=table,
            attributes=attributes,
            where_clause=where_clause,
        )

        raw_first, elapsed, iter_tokens = self._call_llm(first_prompt)
        total_time += elapsed
        total_tokens += iter_tokens

        content = self._get_content(raw_first)
        parsed_first = ResponseParser.parse_llm_response(content, elapsed, iter_tokens)
        rows = parsed_first["result_set"]

        # Deduplicate
        new_rows = _filter_new_rows(rows, seen)
        all_rows.extend(new_rows)

        # If first iteration yields nothing → stop immediately
        if not all_rows:
            return {
                "result_set": [],
                "time": round(total_time, 3),
                "tokens": total_tokens,
                "meta": {
                    "iterations_used": 1,
                    "strategy": "table_scan",
                },
            }

        iterations_used = 1  # first iteration counted

        # --- FOLLOW-UP ITERATIONS ---
        for i in range(1, self.max_iters):
            # build context-based follow-up
            followup_prompt = PromptTemplates.table_scan_followup(previous_rows=all_rows)

            raw_next, elapsed, iter_tokens = self._call_llm(followup_prompt)
            total_time += elapsed
            total_tokens += iter_tokens

            content = self._get_content(raw_next)
            parsed_next = ResponseParser.parse_llm_response(content, elapsed, iter_tokens)
            rows_next = parsed_next["result_set"]

            only_new = _filter_new_rows(rows_next, seen)

            # Stop if no new rows appear
            if not only_new:
                break

            all_rows.extend(only_new)
            iterations_used += 1

        return {
            "result_set": all_rows,
            "time": round(total_time, 3),
            "tokens": total_tokens,
            "meta": {
                "iterations_used": iterations_used,
                "strategy": "table_scan",
            },
        }

