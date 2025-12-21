"""
Galois_A: Push ALL conditions
Simplest optimization - pushes entire WHERE clause
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.galois.operators.base_scan import project_rows


from ..operators.table_scan import TableScanOperator
from ..utils.where_columns import extract_where_columns
from ..utils.post_filter import post_filter_rows_duckdb


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
        extended_attrs = attributes
        if where_clause:
            where_cols = extract_where_columns(where_clause)
            # preserve order: original attributes first
            extended_attrs = attributes + [c for c in where_cols if c not in attributes]
        result = self.table_scan.scan(
            dataset=dataset,
            table=table,
            attributes=extended_attrs,
            where_clause=where_clause,   # full pushdown to LLM
        )

        # post-filtering (in case LLM made mistakes)
        if where_clause:
            result["result_set"] = post_filter_rows_duckdb(result["result_set"], [where_clause])

        # project back
        result["result_set"] = project_rows(result["result_set"], attributes)

        meta = result.setdefault("meta", {})
        meta["variant"] = "Galois_A"
        meta["pushdown"] = "all"

        return result
