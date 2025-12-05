"""
Galois_A: Push ALL conditions
Simplest optimization - pushes entire WHERE clause
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..operators.table_scan import TableScanOperator
from ..utils.where_evaluator import WhereEvaluator


class GaloisA:
    """
    Galois_A: Push ALL conditions into LLMScan.
    
    Strategy:
    - Uses Table-Scan operator
    - Pushes entire WHERE clause down
    - Cheapest in terms of tokens
    - May reduce quality if WHERE is too complex
    
    This is the simplest optimization and serves as a good baseline.
    """

    def __init__(self, llm, max_iters: int = 5):
        """
        Args:
            llm: LangChain ChatModel instance
            max_iters: Maximum iterations for Table-Scan
        """
        self.table_scan = TableScanOperator(llm, max_iters)

    def execute(
        self,
        dataset: str,
        table: str,
        attributes: List[str],
        where_clause: Optional[str] = None,
    ) -> Dict[str, Any]:
        result = self.table_scan.scan(
            dataset=dataset,
            table=table,
            attributes=attributes,
            where_clause=where_clause,   # full pushdown to LLM
        )

        # post-filtering (in case LLM made mistakes)
        if where_clause:
            result["result_set"] = WhereEvaluator.filter_rows(
                result["result_set"], where_clause
            )

        meta = result.setdefault("meta", {})
        meta["variant"] = "Galois_A"
        meta["pushdown"] = "all"

        return result
