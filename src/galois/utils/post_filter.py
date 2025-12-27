from __future__ import annotations

from typing import Any, Dict, List, Optional

import duckdb

def post_filter_rows_duckdb(
    rows: List[Dict[str, Any]],
    residual_conditions: Optional[List[str]],
    *,
    view_name: str = "virtual_table",
) -> List[Dict[str, Any]]:
    """
    Apply residual WHERE conditions locally using DuckDB.

    - rows: list of dicts from LLM
    - residual_conditions: list of SQL predicate strings (e.g. ["a=1", "name='X'"])
    """
    if not rows:
        return []
    if not residual_conditions:
        return rows

    combined = " AND ".join([c.strip() for c in residual_conditions if c and c.strip()])
    if not combined:
        return rows

    con = duckdb.connect(database=":memory:")
    try:
        # DuckDB can register a list[dict] directly as a relation in recent versions.
        # If that is unsupported in this runtime, replace the registration with a pandas.DataFrame.
        con.register(view_name, rows)
        rel = con.table(view_name)
        filtered = rel.filter(combined).df()
        return filtered.to_dict(orient="records")
    except Exception:
        # Fail-open: return unfiltered rows if DuckDB cannot evaluate condition
        return rows
    finally:
        con.close()
