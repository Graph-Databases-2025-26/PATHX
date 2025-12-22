from __future__ import annotations

from typing import Set

from sqlglot import parse_one, exp


def extract_where_columns(where_clause: str) -> Set[str]:
    """
    Extract column names referenced in a WHERE clause.

    Returns a set like {"airline", "call_sign", "elevation_in_ft"}.

    This is intentionally lightweight and does NOT evaluate the WHERE.
    """
    if not where_clause or not where_clause.strip():
        return set()

    # Build a minimal SQL wrapper so sqlglot can parse expressions
    sql = f"SELECT * FROM t WHERE {where_clause}"

    try:
        parsed = parse_one(sql, read="duckdb")
    except Exception:
        # Conservative fallback: return empty set (no extension)
        return set()

    cols: Set[str] = set()
    for c in parsed.find_all(exp.Column):
        # c.name is the column name without table qualifier
        if c.name:
            cols.add(c.name)

    return cols
