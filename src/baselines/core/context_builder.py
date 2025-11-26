from __future__ import annotations
from typing import Dict, Any

from ...db_interaction.duckdb_client import get_duckdb_for_schema


def get_db_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    payload: {
      "database": str,
      "b_type": "SQL" or "NL",
      "sql": str or None,
      "nl_question": str or None,
      "strategy": str,
    }
    """
    database = payload["database"]
    b_type = payload["b_type"]
    sql = payload.get("sql")
    nl_question = payload.get("nl_question")

    db = get_duckdb_for_schema(database)
    schema_info = db.get_table_info()

    return {
        "dataset": database,
        "schema_info": schema_info,
        "sql": sql,
        "nl_question": nl_question,
        "strategy": payload["strategy"],
    }
