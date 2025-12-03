"""
Galois_A: Push ALL conditions
Simplest optimization - pushes entire WHERE clause
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..operators.table_scan import TableScanOperator


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
        """
        Execute query with push-all strategy.
        
        Simply delegates to Table-Scan with full WHERE clause.
        """
        result = self.table_scan.scan(
            dataset=dataset,
            table=table,
            attributes=attributes,
            where_clause=where_clause,  # Push everything
        )
        
        # Add strategy metadata
        result["meta"]["variant"] = "Galois_A"
        result["meta"]["pushdown"] = "all"
        
        return result
