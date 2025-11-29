from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Literal
from enum import Enum

from src.galois.llm_scan import LLMScanOperator


class PushdownStrategy(str, Enum):
    """Strategy for pushing down WHERE predicates to LLMScan"""
    NONE = "none"           # No pushdown - filter after scan
    SINGLE = "single"       # Push down one predicate only
    ALL = "all"            # Push down all predicates


class FilterLLMScanOperator(LLMScanOperator):
    """
    Extended LLMScan operator with WHERE predicate pushdown support.
    
    Implements three logical pushdown strategies:
    - NONE: Execute full table scan, apply filters in post-processing
    - SINGLE: Push down the most selective predicate to LLM prompt
    - ALL: Push down all predicates to LLM prompt
    
    Inherits from LLMScanOperator to reuse core scanning logic.
    """

    def __init__(
        self, 
        llm, 
        max_iters: int = 5,
        pushdown_strategy: PushdownStrategy = PushdownStrategy.ALL
    ):
        """
        Args:
            llm: LangChain ChatModel instance
            max_iters: Maximum number of follow-up iterations
            pushdown_strategy: Which predicates to push down to the LLM
        """
        super().__init__(llm, max_iters)
        self.pushdown_strategy = pushdown_strategy

    def apply_pushdown(
        self,
        dataset: str,
        table: str,
        attributes: List[str],
        where_clause: Optional[str] = None,
        strategy: Optional[PushdownStrategy] = None,
    ) -> Dict[str, Any]:
        """
        Apply pushdown strategy and execute filtered table scan.
        
        Args:
            dataset: Dataset name (e.g., "WORLD")
            table: Table name (e.g., "city")
            attributes: List of column names to retrieve
            where_clause: SQL WHERE clause (without the "WHERE" keyword)
            strategy: Override the instance's default pushdown strategy
            
        Returns:
            Dictionary with:
                - result_set: List of rows (possibly filtered)
                - time: Total execution time
                - tokens: Total tokens used
                - meta: Metadata including pushdown info
        """
        # Use provided strategy or fall back to instance default
        active_strategy = strategy or self.pushdown_strategy
        
        # Parse predicates from WHERE clause
        predicates = self._parse_where_clause(where_clause) if where_clause else []
        
        # Determine which predicates to push down
        pushed_predicates, post_filter_predicates = self._split_predicates(
            predicates, active_strategy
        )
        
        # Build WHERE clause for LLM prompt
        llm_where_clause = self._build_where_clause(pushed_predicates) if pushed_predicates else None
        
        # Execute table scan with pushed-down predicates
        result = self.table_scan(
            dataset=dataset,
            table=table,
            attributes=attributes,
            where_clause=llm_where_clause,
        )
        
        # Apply post-filtering for predicates not pushed down
        if post_filter_predicates:
            result["result_set"] = self._apply_post_filter(
                result["result_set"], 
                post_filter_predicates
            )
        
        # Add pushdown metadata
        result["meta"]["pushdown_strategy"] = active_strategy.value
        result["meta"]["pushed_predicates"] = len(pushed_predicates)
        result["meta"]["post_filter_predicates"] = len(post_filter_predicates)
        
        return result

    def _parse_where_clause(self, where_clause: str) -> List[Dict[str, Any]]:
        """
        Parse SQL WHERE clause into structured predicates.
        
        Handles common operators: =, !=, <, >, <=, >=, LIKE, IN
        Supports AND/OR logic (splits on AND for simplicity)
        
        Args:
            where_clause: SQL WHERE clause string
            
        Returns:
            List of predicate dictionaries with keys:
                - column: Column name
                - operator: Comparison operator
                - value: Comparison value
                - raw: Original predicate string
        """
        if not where_clause:
            return []
        
        predicates = []
        
        # Split on AND (case-insensitive)
        # Note: This simple approach doesn't handle nested OR/AND logic
        parts = re.split(r'\s+AND\s+', where_clause, flags=re.IGNORECASE)
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
            
            predicate = self._parse_single_predicate(part)
            if predicate:
                predicates.append(predicate)
        
        return predicates

    def _parse_single_predicate(self, predicate_str: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single predicate into structured format.
        
        Examples:
            "Population > 1000000" -> {column: "Population", operator: ">", value: 1000000}
            "Name LIKE 'New%'" -> {column: "Name", operator: "LIKE", value: "New%"}
            "CountryCode = 'USA'" -> {column: "CountryCode", operator: "=", value: "USA"}
        """
        # Pattern for standard comparisons: column operator value
        patterns = [
            # LIKE operator
            (r"(\w+)\s+LIKE\s+'([^']+)'", "LIKE"),
            # IN operator
            (r"(\w+)\s+IN\s*\(([^)]+)\)", "IN"),
            # Comparison operators (order matters: check >= before >)
            (r"(\w+)\s*(>=|<=|!=|<>|=|>|<)\s*'([^']+)'", None),  # String values
            (r"(\w+)\s*(>=|<=|!=|<>|=|>|<)\s*(\d+\.?\d*)", None),  # Numeric values
        ]
        
        for pattern, fixed_op in patterns:
            match = re.match(pattern, predicate_str.strip(), re.IGNORECASE)
            if match:
                groups = match.groups()
                column = groups[0]
                
                if fixed_op:
                    operator = fixed_op
                    value = groups[1]
                else:
                    operator = groups[1]
                    value = groups[2]
                
                # Type conversion for numeric values
                if operator not in ("LIKE", "IN") and value.replace(".", "").isdigit():
                    value = float(value) if "." in value else int(value)
                
                # Handle IN operator - convert to list
                if operator == "IN":
                    value = [v.strip().strip("'\"") for v in value.split(",")]
                
                return {
                    "column": column,
                    "operator": operator,
                    "value": value,
                    "raw": predicate_str.strip(),
                }
        
        # If no pattern matched, return raw predicate
        return {
            "column": None,
            "operator": None,
            "value": None,
            "raw": predicate_str.strip(),
        }

    def _split_predicates(
        self, 
        predicates: List[Dict[str, Any]], 
        strategy: PushdownStrategy
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Split predicates into pushed-down vs post-filter based on strategy.
        
        Args:
            predicates: List of parsed predicates
            strategy: Pushdown strategy to apply
            
        Returns:
            Tuple of (pushed_predicates, post_filter_predicates)
        """
        if strategy == PushdownStrategy.NONE:
            return [], predicates
        
        elif strategy == PushdownStrategy.SINGLE:
            # Push down the first predicate (could be made smarter with selectivity)
            if predicates:
                return [predicates[0]], predicates[1:]
            return [], []
        
        elif strategy == PushdownStrategy.ALL:
            return predicates, []
        
        return [], predicates

    def _build_where_clause(self, predicates: List[Dict[str, Any]]) -> str:
        """
        Build WHERE clause string from structured predicates.
        
        Args:
            predicates: List of predicate dictionaries
            
        Returns:
            WHERE clause string (without "WHERE" keyword)
        """
        if not predicates:
            return ""
        
        clause_parts = []
        for pred in predicates:
            # Use raw string if available, otherwise reconstruct
            if pred.get("raw"):
                clause_parts.append(pred["raw"])
            elif pred.get("column") and pred.get("operator"):
                column = pred["column"]
                operator = pred["operator"]
                value = pred["value"]
                
                if operator == "IN":
                    values_str = ", ".join(f"'{v}'" for v in value)
                    clause_parts.append(f"{column} IN ({values_str})")
                elif isinstance(value, str):
                    clause_parts.append(f"{column} {operator} '{value}'")
                else:
                    clause_parts.append(f"{column} {operator} {value}")
        
        return " AND ".join(clause_parts)

    def _apply_post_filter(
        self, 
        rows: List[Dict[str, Any]], 
        predicates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply predicates as post-processing filters on result set.
        
        Args:
            rows: List of row dictionaries
            predicates: List of predicates to apply
            
        Returns:
            Filtered list of rows
        """
        filtered_rows = rows
        
        for pred in predicates:
            filtered_rows = [
                row for row in filtered_rows 
                if self._evaluate_predicate(row, pred)
            ]
        
        return filtered_rows

    def _evaluate_predicate(
        self, 
        row: Dict[str, Any], 
        predicate: Dict[str, Any]
    ) -> bool:
        """
        Evaluate a single predicate against a row.
        
        Args:
            row: Row dictionary
            predicate: Predicate dictionary
            
        Returns:
            True if row satisfies predicate, False otherwise
        """
        column = predicate.get("column")
        operator = predicate.get("operator")
        value = predicate.get("value")
        
        # Skip if predicate is malformed
        if not column or not operator:
            return True
        
        # Get row value (case-insensitive column lookup)
        row_value = None
        for key in row.keys():
            if key.lower() == column.lower():
                row_value = row[key]
                break
        
        if row_value is None:
            return False
        
        # Evaluate based on operator
        try:
            if operator == "=":
                return row_value == value
            elif operator in ("!=", "<>"):
                return row_value != value
            elif operator == ">":
                return float(row_value) > float(value)
            elif operator == "<":
                return float(row_value) < float(value)
            elif operator == ">=":
                return float(row_value) >= float(value)
            elif operator == "<=":
                return float(row_value) <= float(value)
            elif operator == "LIKE":
                # Simple LIKE implementation (% as wildcard)
                pattern = value.replace("%", ".*")
                return re.match(f"^{pattern}$", str(row_value), re.IGNORECASE) is not None
            elif operator == "IN":
                return row_value in value
            else:
                return True
        except (ValueError, TypeError):
            return False