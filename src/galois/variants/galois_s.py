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
from ..utils.where_evaluator import WhereEvaluator
import time
from ..operators.base_scan import extract_total_tokens

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

        # Parse conditions
        conditions = PushdownOptimizer.parse_conditions(where_clause)

        # CASE 1 — zero or one condition
        if len(conditions) <= 1:
            result = self.table_scan.scan(
                dataset=dataset,
                table=table,
                attributes=attributes,
                where_clause=where_clause,
            )

            # Post-filter using the full WHERE clause
            if where_clause:
                result["result_set"] = WhereEvaluator.filter_rows(result["result_set"], where_clause)

            meta = result.setdefault("meta", {})
            meta["variant"] = "Galois_S"
            meta["pushdown"] = "all" if where_clause else "none"
            meta["selectivity_evaluation"] = {cond: "higher" for cond in conditions} if conditions else {}
            meta["pushed_conditions"] = where_clause

            return result

        # CASE 2 — multiple conditions; call LLM to evaluate selectivity
        selectivity_map, sel_time, sel_tokens = self._evaluate_selectivity(
            dataset, table, conditions
        )

        # determine pushdown
        pushdown_clause = PushdownOptimizer.determine_pushdown(
            conditions=conditions,
            evaluation=selectivity_map,
            mode="selectivity",
        )

        # run actual scan
        result = self.table_scan.scan(
            dataset=dataset,
            table=table,
            attributes=attributes,
            where_clause=pushdown_clause,
        )

        # add planning cost
        result["time"] += sel_time
        result["tokens"] += sel_tokens

        # post-filter full WHERE
        if where_clause:
            result["result_set"] = WhereEvaluator.filter_rows(result["result_set"], where_clause)

        # metadata
        meta = result.setdefault("meta", {})
        meta["variant"] = "Galois_S"
        meta["pushdown"] = "selective"
        meta["selectivity_evaluation"] = selectivity_map
        meta["pushed_conditions"] = pushdown_clause

        return result


    def _evaluate_selectivity(
        self,
        dataset: str,
        table: str,
        conditions: List[str],
    ) -> tuple[Dict[str, str], float, int]:
        """
        Ask LLM to evaluate selectivity of each condition.
        Returns (selectivity_map, elapsed_time, token_count).
        """
        prompt = PromptTemplates.evaluate_selectivity(dataset, table, conditions)

        try:
            t0 = time.time()
            raw = self._chain.invoke({"prompt": prompt})
            elapsed = time.time() - t0
            tokens = extract_total_tokens(raw)

            content = raw.content if hasattr(raw, "content") else str(raw)
            response_json = ResponseParser.extract_json(content)
            selectivity_map = ResponseParser.extract_selectivity_map(response_json)

            # Ensure all conditions have evaluation
            for cond in conditions:
                if cond not in selectivity_map:
                    selectivity_map[cond] = "lower"

            return selectivity_map, elapsed, tokens

        except Exception as e:
            print(f"Warning: Failed to evaluate selectivity: {e}")
            return ({cond: "lower" for cond in conditions}, 0.0, 0)

