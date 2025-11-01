"""
Centralized logging configuration for pathX.

Provides:
- `logger`: a configured global logger.
- `log_query_event(event_type, sql)`: lightweight structured query log entry.
"""

import logging
import sys
import os
from datetime import datetime

# --- Basic configuration ---
LOG_LEVEL = os.getenv("PATHX_LOG_LEVEL", "INFO").upper()

logger = logging.getLogger("pathx")
if not logger.hasHandlers():
    logger.setLevel(LOG_LEVEL)
    handler = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False


# --- Helper for SQL / explain events ---
def log_query_event(event_type: str, sql: str):
    """
    Record a short log entry for query-related operations (EXPLAIN, EXECUTE, etc.)
    """
    sql_short = " ".join(sql.strip().split())
    if len(sql_short) > 120:
        sql_short = sql_short[:117] + "..."
    logger.debug(f"[{event_type}] SQL={sql_short}")


# --- Example of usage ---
if __name__ == "__main__":
    logger.info("Logger initialized successfully.")
    log_query_event("EXPLAIN", "SELECT * FROM target.movies WHERE startyear > 2000")
