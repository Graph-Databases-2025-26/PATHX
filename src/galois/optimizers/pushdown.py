"""
Pushdown optimization logic for WHERE clause conditions
"""
from __future__ import annotations

import re
from typing import List, Optional


class PushdownOptimizer:
    """
    Utilities for parsing and optimizing WHERE clause pushdown.
    
    Supports:
    - Parsing conditions from WHERE clause
    - Reconstructing WHERE clause from selected conditions
    """

    @staticmethod
    def parse_conditions(where_clause: Optional[str]) -> List[str]:
        """
        Parse WHERE clause into individual conditions.
        
        Examples:
            "population > 1M AND num_hospitals > 15"
            -> ["population > 1M", "num_hospitals > 15"]
            
            "area > 300K"
            -> ["area > 300K"]
        
        Note: This is a simplified parser that splits on AND.
        A production version would handle:
        - OR operators
        - Nested conditions with parentheses
        - Complex predicates
        """
        if not where_clause:
            return []
        
        # Simple split on AND (case-insensitive)
        # This doesn't handle nested parentheses or OR
        conditions = []
        
        # Split on AND, preserving spaces
        parts = re.split(r'\s+AND\s+', where_clause, flags=re.IGNORECASE)
        
        for part in parts:
            part = part.strip()
            if part:
                conditions.append(part)
        
        return conditions

    @staticmethod
    def build_where_clause(conditions: List[str]) -> Optional[str]:
        """
        Reconstruct WHERE clause from list of conditions.
        
        Args:
            conditions: List of condition strings
        
        Returns:
            Combined WHERE clause (without "WHERE" keyword)
            None if conditions is empty
        """
        if not conditions:
            return None
        
        return " AND ".join(conditions)

    @staticmethod
    def select_conditions(
        all_conditions: List[str],
        evaluation: dict[str, str],
        criterion: str = "high",
    ) -> List[str]:
        """
        Select conditions based on evaluation results.
        
        Args:
            all_conditions: All available conditions
            evaluation: Dict mapping condition -> evaluation value
            criterion: What value to filter for (e.g., "high", "higher")
        
        Returns:
            List of conditions matching the criterion
        """
        selected = []
        for cond in all_conditions:
            # Try exact match first
            if cond in evaluation and evaluation[cond] == criterion:
                selected.append(cond)
            else:
                # Try normalized match (strip whitespace, case-insensitive)
                norm_cond = cond.strip().lower()
                for eval_cond, eval_value in evaluation.items():
                    if eval_cond.strip().lower() == norm_cond:
                        if eval_value == criterion:
                            selected.append(cond)
                        break
        
        return selected

    @staticmethod
    def determine_pushdown(
        conditions: List[str],
        evaluation: dict[str, str],
        mode: str = "confidence",
    ) -> Optional[str]:
        """
        Determine which conditions to push down based on evaluation.
        
        Implements the logic from the paper:
        - If only one condition is "high"/"higher", push it
        - If multiple are "high"/"higher", push all
        - If none are "high"/"higher", push nothing
        
        Args:
            conditions: All conditions from WHERE clause
            evaluation: Dict with evaluation results
            mode: "confidence" (high/low) or "selectivity" (higher/lower)
        
        Returns:
            WHERE clause to push down, or None
        """
        criterion = "high" if mode == "confidence" else "higher"
        
        selected = PushdownOptimizer.select_conditions(
            all_conditions=conditions,
            evaluation=evaluation,
            criterion=criterion,
        )
        
        if len(selected) == 0:
            # No conditions meet criterion - don't push
            return None
        elif len(selected) == 1:
            # Exactly one - push it
            return selected[0]
        else:
            # Multiple - push all of them
            return PushdownOptimizer.build_where_clause(selected)
