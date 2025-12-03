"""
Galois_S: Push SELECTIVE condition
Uses LLM to evaluate selectivity and pushes accordingly
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..operators.table_scan import TableScanOperator
from ..optimizers.pushdown import PushdownOptimizer
from ..prompts.prompt_templates import PromptTemplates
from ..utils.response_parser import ResponseParser


class GaloisS:
    """
    Galois_S: Push most SELECTIVE condition(s).
    
    Strategy:
    1. Use LLM to evaluate selectivity of each condition
    2. Push based on evaluation:
       - If 1 condition is "higher": push it
       - If >1 are "higher": push all
       - If 0 are "higher": push nothing
    3. Use Table-Scan with selected pushdown
    
    This simulates traditional DBMS optimization based on selectivity.
    """

    def __init__(self, llm, max_iters: int = 5):
        """
        Args:
            llm: LangChain ChatModel instance
            max_iters: Maximum iterations for Table-Scan
        """
        self.llm = llm
        self.table_scan = TableScanOperator(llm, max_iters)
        
        # Setup simple prompt chain for selectivity evaluation
        from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
        chat_prompt = ChatPromptTemplate.from_messages([
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=["prompt"],
                    template="{prompt}",
                )
            )
        ])
        self._chain = chat_prompt | llm

    def execute(
        self,
        dataset: str,
        table: str,
        attributes: List[str],
        where_clause: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute query with selectivity-based pushdown.
        """
        # Parse conditions
        conditions = PushdownOptimizer.parse_conditions(where_clause)
        
        if len(conditions) <= 1:
            # 0 or 1 condition: just push it (or nothing)
            return self.table_scan.scan(dataset, table, attributes, where_clause)
        
        # Multiple conditions: evaluate selectivity
        selectivity_map = self._evaluate_selectivity(dataset, table, conditions)
        
        # Determine what to push
        pushdown_clause = PushdownOptimizer.determine_pushdown(
            conditions=conditions,
            evaluation=selectivity_map,
            mode="selectivity",
        )
        
        # Execute with selected pushdown
        result = self.table_scan.scan(
            dataset=dataset,
            table=table,
            attributes=attributes,
            where_clause=pushdown_clause,
        )
        
        # Add metadata
        result["meta"]["variant"] = "Galois_S"
        result["meta"]["pushdown"] = "selective"
        result["meta"]["selectivity_evaluation"] = selectivity_map
        result["meta"]["pushed_conditions"] = pushdown_clause
        
        return result

    def _evaluate_selectivity(
        self,
        dataset: str,
        table: str,
        conditions: List[str],
    ) -> Dict[str, str]:
        """
        Ask LLM to evaluate selectivity of each condition.
        
        Returns:
            Dict mapping condition -> "higher" or "lower"
        """
        prompt = PromptTemplates.evaluate_selectivity(dataset, table, conditions)
        
        try:
            raw = self._chain.invoke({"prompt": prompt})
            content = raw.content if hasattr(raw, "content") else str(raw)
            
            response_json = ResponseParser.extract_json(content)
            selectivity_map = ResponseParser.extract_selectivity_map(response_json)
            
            # Ensure all conditions have an evaluation
            for cond in conditions:
                if cond not in selectivity_map:
                    selectivity_map[cond] = "lower"  # Default to lower
            
            return selectivity_map
        
        except Exception as e:
            print(f"Warning: Failed to evaluate selectivity: {e}")
            # Fallback: treat all as "lower" (no pushdown)
            return {cond: "lower" for cond in conditions}
