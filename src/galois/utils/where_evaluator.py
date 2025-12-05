"""
WHERE clause evaluator for filtering rows in memory
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Callable
from operator import eq, ne, lt, le, gt, ge


class WhereEvaluator:
    """
    Evaluates SQL WHERE clauses against rows in memory.
    
    Supports:
    - Comparison operators: =, !=, <, <=, >, >=
    - Logical operators: AND, OR
    - Parentheses for grouping
    - String comparisons (case-insensitive)
    - Numeric comparisons
    - NULL handling (IS NULL, IS NOT NULL)
    
    Limitations:
    - LIKE patterns: basic % wildcard only
    - IN operator: simple list format only
    - No subqueries
    """

    # Operator mappings (ordered to avoid '=' catching '>=' / '<=' etc.)
    OPERATORS = [
        ('>=', ge),
        ('<=', le),
        ('<>', ne),
        ('!=', ne),
        ('=', eq),
        ('>', gt),
        ('<', lt),
    ]

    def _evaluate_comparison(self, condition: str, row: Dict[str, Any]) -> bool:
        """Evaluate comparison operators (=, !=, <, >, etc.)"""

        cond = condition.strip()

        # Try each operator, in order (longest first)
        for op_str, op_func in self.OPERATORS:
            if op_str in cond:
                parts = cond.split(op_str, 1)
                if len(parts) != 2:
                    continue

                left = parts[0].strip()
                right = parts[1].strip()

                # Get values
                left_val = self._get_value(left, row)
                right_val = self._parse_literal(right)

                # Handle NULL comparisons (we just treat them as False here;
                # IS NULL / IS NOT NULL обрабатываются в _evaluate_condition)
                if left_val is None or right_val is None:
                    return False

                # Type coercion for comparison
                left_val, right_val = self._coerce_types(left_val, right_val)

                try:
                    return op_func(left_val, right_val)
                except Exception:
                    return False

        # No operator found - treat as boolean attribute
        attr = cond
        val = row.get(attr)
        return bool(val)

    @staticmethod
    def filter_rows(rows: List[Dict[str, Any]], where_clause: str) -> List[Dict[str, Any]]:
        """
        Filter rows based on WHERE clause.
        
        Args:
            rows: List of row dictionaries
            where_clause: SQL WHERE clause (without "WHERE" keyword)
        
        Returns:
            Filtered list of rows
        """
        if not where_clause or not where_clause.strip():
            return rows

        evaluator = WhereEvaluator()
        filtered = []

        for row in rows:
            try:
                if evaluator.evaluate(where_clause, row):
                    filtered.append(row)
            except Exception as e:
                # If evaluation fails, exclude row and log warning
                print(f"Warning: Failed to evaluate WHERE for row {row}: {e}")

        return filtered

    def evaluate(self, where_clause: str, row: Dict[str, Any]) -> bool:
        """
        Evaluate WHERE clause against a single row.
        
        Returns:
            True if row matches condition, False otherwise
        """
        # Normalize the clause
        clause = where_clause.strip()
        
        # Handle logical operators (AND, OR)
        return self._evaluate_logical(clause, row)

    def _evaluate_logical(self, clause: str, row: Dict[str, Any]) -> bool:
        """Handle AND/OR logical operators"""
        
        # Split by OR (lowest precedence)
        or_parts = self._split_by_operator(clause, 'OR')
        if len(or_parts) > 1:
            return any(self._evaluate_and(part, row) for part in or_parts)
        
        # No OR found, evaluate as AND expression
        return self._evaluate_and(clause, row)

    def _evaluate_and(self, clause: str, row: Dict[str, Any]) -> bool:
        """Handle AND operator"""
        
        # Split by AND
        and_parts = self._split_by_operator(clause, 'AND')
        if len(and_parts) > 1:
            return all(self._evaluate_condition(part, row) for part in and_parts)
        
        # No AND found, evaluate as single condition
        return self._evaluate_condition(clause, row)

    def _evaluate_condition(self, condition: str, row: Dict[str, Any]) -> bool:
        """Evaluate a single condition"""
        
        condition = condition.strip()
        
        # Remove outer parentheses if present
        if condition.startswith('(') and condition.endswith(')'):
            # Check if parentheses are balanced
            if self._are_outer_parentheses(condition):
                return self._evaluate_logical(condition[1:-1], row)
        
        # Handle IS NULL / IS NOT NULL
        upper = condition.upper()

        if 'IS NOT NULL' in upper:
            # use original condition to preserve column name casing
            attr_expr = re.split(r'\s+IS\s+NOT\s+NULL', condition, flags=re.IGNORECASE)[0].strip()
            val = self._get_value(attr_expr, row)
            return val is not None
        
        if 'IS NULL' in upper:
            attr_expr = re.split(r'\s+IS\s+NULL', condition, flags=re.IGNORECASE)[0].strip()
            val = self._get_value(attr_expr, row)
            return val is None

        
        # Handle LIKE
        if ' LIKE ' in condition.upper():
            return self._evaluate_like(condition, row)
        
        # Handle IN
        if ' IN ' in condition.upper():
            return self._evaluate_in(condition, row)
        
        # Handle comparison operators
        return self._evaluate_comparison(condition, row)

    def _evaluate_like(self, condition: str, row: Dict[str, Any]) -> bool:
        """Evaluate LIKE pattern matching"""
        
        parts = re.split(r'\s+LIKE\s+', condition, flags=re.IGNORECASE)
        if len(parts) != 2:
            return False
        
        attr = parts[0].strip()
        pattern = parts[1].strip().strip("'\"")
        
        value = self._get_value(attr, row)
        if value is None:
            return False
        
        # Convert to string
        value_str = str(value).lower()
        
        # Simple LIKE: % is wildcard
        pattern_regex = pattern.lower().replace('%', '.*')
        pattern_regex = f'^{pattern_regex}$'
        
        return bool(re.match(pattern_regex, value_str))

    def _evaluate_in(self, condition: str, row: Dict[str, Any]) -> bool:
        """Evaluate IN operator"""
        
        parts = re.split(r'\s+IN\s+', condition, flags=re.IGNORECASE)
        if len(parts) != 2:
            return False
        
        attr = parts[0].strip()
        values_str = parts[1].strip()
        
        # Parse list: IN ('val1', 'val2', 123)
        values_str = values_str.strip('()')
        values = [self._parse_literal(v.strip()) for v in values_str.split(',')]
        
        row_value = self._get_value(attr, row)
        
        # Try exact match first
        if row_value in values:
            return True
        
        # Try with type coercion
        for val in values:
            try:
                coerced_row, coerced_val = self._coerce_types(row_value, val)
                if coerced_row == coerced_val:
                    return True
            except Exception:
                continue
        
        return False

    def _get_value(self, expr: str, row: Dict[str, Any]) -> Any:
        """Get value from row or parse as literal"""
        
        expr = expr.strip()
        
        # Check if it's a column name (exists in row)
        if expr in row:
            return row[expr]
        
        # Try case-insensitive match
        for key, value in row.items():
            if key.lower() == expr.lower():
                return value
        
        # Not a column, parse as literal
        return self._parse_literal(expr)

    def _parse_literal(self, value: str) -> Any:
        """Parse a literal value (string, number, etc.)"""
        
        value = value.strip()
        
        # String literal (quoted)
        if (value.startswith("'") and value.endswith("'")) or \
           (value.startswith('"') and value.endswith('"')):
            return value[1:-1]
        
        # NULL
        if value.upper() == 'NULL':
            return None
        
        # Boolean
        if value.upper() == 'TRUE':
            return True
        if value.upper() == 'FALSE':
            return False
        
        # Number with units (e.g., "1M", "5K")
        if re.match(r'^\d+(\.\d+)?[KMB]$', value.upper()):
            return self._parse_number_with_unit(value)
        
        # Try integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Try float
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value

    def _parse_number_with_unit(self, value: str) -> float:
        """Parse numbers like 1M, 5K, 2B"""
        
        value = value.upper()
        units = {'K': 1000, 'M': 1000000, 'B': 1000000000}
        
        for unit, multiplier in units.items():
            if value.endswith(unit):
                num_str = value[:-1]
                try:
                    return float(num_str) * multiplier
                except ValueError:
                    return 0
        
        return 0

    def _coerce_types(self, left: Any, right: Any) -> tuple[Any, Any]:
        """Coerce types for comparison"""
        
        # Both None
        if left is None and right is None:
            return None, None
        
        # One is None
        if left is None or right is None:
            return left, right
        
        # Both same type
        if type(left) == type(right):
            return left, right
        
        # Try numeric coercion
        try:
            if isinstance(left, (int, float)) or isinstance(right, (int, float)):
                return float(left), float(right)
        except (ValueError, TypeError):
            pass
        
        # String comparison (case-insensitive)
        return str(left).lower(), str(right).lower()

    def _split_by_operator(self, clause: str, operator: str) -> List[str]:
        """Split clause by logical operator, respecting parentheses"""
        
        parts = []
        current = []
        depth = 0
        
        # Split by spaces to find operator
        tokens = clause.split()
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            # Track parentheses depth
            depth += token.count('(') - token.count(')')
            
            # Check if this is the operator at depth 0
            if depth == 0 and token.upper() == operator.upper():
                # Found operator - save current part
                if current:
                    parts.append(' '.join(current))
                    current = []
            else:
                current.append(token)
            
            i += 1
        
        # Add final part
        if current:
            parts.append(' '.join(current))
        
        return parts if len(parts) > 1 else [clause]

    def _are_outer_parentheses(self, s: str) -> bool:
        """Check if outer parentheses are balanced and enclosing"""
        
        if not (s.startswith('(') and s.endswith(')')):
            return False
        
        depth = 0
        for i, char in enumerate(s):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            
            # If depth hits 0 before the end, outer parens don't enclose all
            if depth == 0 and i < len(s) - 1:
                return False
        
        return depth == 0
