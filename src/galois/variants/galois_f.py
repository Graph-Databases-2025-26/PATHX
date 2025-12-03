"""
Galois_F: FULL optimization
Combines logical (confidence-based pushdown) and physical (scan selection) optimizations
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..operators.table_scan import TableScanOperator
from ..operators.key_scan import KeyScanOperator
from ..optimizers.pushdown import PushdownOptimizer
from ..prompts.prompt_templates import PromptTemplates
from ..utils.response_parser import ResponseParser


class GaloisF:
    """
    Galois_F: FULL optimization with both logical and physical plans.
    
    Strategy:
    1. LOGICAL: Evaluate confidence for each condition, push based on "high" confidence
    2. PHYSICAL: Estimate confidence for key retrieval, choose scan strategy:
       - If conf(q) >= threshold: use Key-Scan (more accurate)
       - If conf(q) < threshold: use Table-Scan (cheaper)
    
    Formula from paper: conf(q) = LLM_conf(keys|conds)^(1/n)
    where n = number of attributes
    
    This achieves the best quality-cost tradeoff.
    """

    def __init__(
        self,
        llm,
        max_iters: int = 5,
        confidence_threshold: float = 0.6,
        key_attributes: Optional[List[str]] = None,
    ):
        """
        Args:
            llm: LangChain ChatModel instance
            max_iters: Maximum iterations
            confidence_threshold: Tau threshold from paper (default 0.6)
            key_attributes: Key attribute names (default: first attribute)
        """
        self.llm = llm
        self.max_iters = max_iters
        self.confidence_threshold = confidence_threshold
        self.key_attributes = key_attributes
        
        self.table_scan = TableScanOperator(llm, max_iters)
        self.key_scan = KeyScanOperator(llm, max_iters, key_attributes)
        
        # Setup prompt chain
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
        Execute query with full optimization.
        
        Step 1: LOGICAL - Determine pushdown based on confidence
        Step 2: PHYSICAL - Choose scan strategy based on confidence
        """
        # Parse conditions
        conditions = PushdownOptimizer.parse_conditions(where_clause)
        
        # STEP 1: LOGICAL OPTIMIZATION
        if len(conditions) <= 1:
            # Single or no condition: push it
            pushdown_clause = where_clause
            confidence_map = {}
        else:
            # Multiple conditions: evaluate confidence
            confidence_map = self._evaluate_confidence(dataset, table, conditions)
            
            pushdown_clause = PushdownOptimizer.determine_pushdown(
                conditions=conditions,
                evaluation=confidence_map,
                mode="confidence",
            )
        
        # STEP 2: PHYSICAL OPTIMIZATION
        # Estimate confidence for key retrieval
        key_confidence = self._estimate_key_confidence(
            dataset, table, attributes, pushdown_clause
        )
        
        # Apply formula: conf(q) = LLM_conf(keys|conds)^(1/n)
        n = len(attributes)
        adjusted_confidence = key_confidence ** (1.0 / n)
        
        # Choose scan strategy
        if adjusted_confidence >= self.confidence_threshold:
            # High confidence: use Key-Scan
            result = self.key_scan.scan(
                dataset=dataset,
                table=table,
                attributes=attributes,
                where_clause=pushdown_clause,
            )
            scan_chosen = "key_scan"
        else:
            # Low confidence: use Table-Scan
            result = self.table_scan.scan(
                dataset=dataset,
                table=table,
                attributes=attributes,
                where_clause=pushdown_clause,
            )
            scan_chosen = "table_scan"
        
        # Add metadata
        result["meta"]["variant"] = "Galois_F"
        result["meta"]["pushdown"] = "confidence"
        result["meta"]["confidence_evaluation"] = confidence_map
        result["meta"]["pushed_conditions"] = pushdown_clause
        result["meta"]["key_confidence"] = key_confidence
        result["meta"]["adjusted_confidence"] = adjusted_confidence
        result["meta"]["confidence_threshold"] = self.confidence_threshold
        result["meta"]["scan_chosen"] = scan_chosen
        
        return result

    def _evaluate_confidence(
        self,
        dataset: str,
        table: str,
        conditions: List[str],
    ) -> Dict[str, str]:
        """
        Evaluate LLM's confidence for each condition.
        
        Returns:
            Dict mapping condition -> "high" or "low"
        """
        prompt = PromptTemplates.evaluate_confidence(dataset, table, conditions)
        
        try:
            raw = self._chain.invoke({"prompt": prompt})
            content = raw.content if hasattr(raw, "content") else str(raw)
            
            response_json = ResponseParser.extract_json(content)
            confidence_map = ResponseParser.extract_confidence_map(response_json)
            
            # Ensure all conditions have evaluation
            for cond in conditions:
                if cond not in confidence_map:
                    confidence_map[cond] = "low"  # Default conservative
            
            return confidence_map
        
        except Exception as e:
            print(f"Warning: Failed to evaluate confidence: {e}")
            return {cond: "low" for cond in conditions}

    def _estimate_key_confidence(
        self,
        dataset: str,
        table: str,
        attributes: List[str],
        where_clause: Optional[str],
    ) -> float:
        """
        Estimate LLM's confidence in retrieving keys.
        
        Returns:
            Float between 0.0 and 1.0
        """
        prompt = PromptTemplates.evaluate_key_confidence(
            dataset, table, attributes, where_clause
        )
        
        try:
            raw = self._chain.invoke({"prompt": prompt})
            content = raw.content if hasattr(raw, "content") else str(raw)
            
            response_json = ResponseParser.extract_json(content)
            confidence = ResponseParser.extract_confidence(response_json)
            
            # Clamp to [0, 1]
            return max(0.0, min(1.0, confidence))
        
        except Exception as e:
            print(f"Warning: Failed to estimate key confidence: {e}")
            return 0.5  # Default moderate confidence
