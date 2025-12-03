"""
Table-Scan Operator - Algorithm 1 from the paper
Retrieves complete tuples iteratively
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from .base_scan import BaseScanOperator, _filter_new_rows, _row_signature
from ..prompts.prompt_templates import PromptTemplates
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
        Execute Table-Scan with iterative retrieval.
        
        Args:
            dataset: Dataset name (e.g., "WORLD")
            table: Table name (e.g., "city")
            attributes: List of attribute names to retrieve
            where_clause: Optional SQL WHERE clause (without "WHERE")
        
        Returns:
            Dict with result_set, time, tokens, and metadata
        """
        all_rows: List[Dict] = []
        seen: Set[Tuple] = set()
        
        total_time = 0.0
        total_tokens = 0
        iterations_used = 0

        # Initial call
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
        new_rows = _filter_new_rows(rows, seen)
        all_rows.extend(new_rows)

        # Follow-up iterations
        for i in range(1, self.max_iters + 1):
            if not all_rows:
                # First iteration returned nothing, no point continuing
                break

            followup_prompt = PromptTemplates.table_scan_followup()
            raw_next, elapsed, iter_tokens = self._call_llm(followup_prompt)
            total_time += elapsed
            total_tokens += iter_tokens

            content = self._get_content(raw_next)
            parsed_next = ResponseParser.parse_llm_response(content, elapsed, iter_tokens)
            rows_next = parsed_next["result_set"]
            only_new = _filter_new_rows(rows_next, seen)

            if not only_new:
                # Stop condition: no new rows
                break

            iterations_used += 1
            all_rows.extend(only_new)

        return {
            "result_set": all_rows,
            "time": round(total_time, 3),
            "tokens": total_tokens,
            "meta": {
                "iterations_used": iterations_used,
                "strategy": "table_scan",
            },
        }
