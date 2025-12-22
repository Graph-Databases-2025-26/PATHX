from __future__ import annotations

from pathlib import Path
from typing import List
import re


DATA_ROOT = Path("data")


def discover_datasets() -> List[str]:
    """
    Returns the list of dataset folder names inside DATA_ROOT.
    Only directories are included.
    """
    if not DATA_ROOT.exists():
        return []

    return sorted(
        d.name
        for d in DATA_ROOT.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    )


def dataset_dir(dataset: str) -> Path:
    """
    Returns the full path to the dataset folder inside DATA_ROOT.
    """
    return DATA_ROOT / dataset


def read_queries(dataset_dir: Path) -> List[str]:
    """
    Loads SQL queries from files named queries_*.sql located inside dataset_dir.
    Dataset format:
        --query1
        SELECT ...;
        --query2
        SELECT ...;
    Returns a list of SQL strings (each starts with SELECT, without the --query header).
    """
    sql_files = sorted(dataset_dir.glob("queries_*.sql"))
    queries: List[str] = []

    for f in sql_files:
        text = f.read_text(encoding="utf-8")

        # Split by --queryN headers (case-insensitive)
        parts = re.split(r"(?im)^--\s*query[^\n]*\n", text)

        # parts[0] — ignore preamble before first --query
        for block in parts[1:]:
            q = block.strip()
            if not q:
                continue

            # Many files have a single query ending with ';'
            # Keep content but remove trailing semicolons safely
            q = q.strip()
            if q.endswith(";"):
                q = q[:-1].strip()

            if q:
                queries.append(q)

    return queries



def read_nl_queries(dataset_dir: Path) -> List[str]:
    """
    Reads natural-language queries from the dataset directory.
    """

    txt_files = sorted(dataset_dir.glob("queries_*.txt"))
    if txt_files:
        queries: List[str] = []
        for f in txt_files:
            text = f.read_text(encoding="utf-8")

            # --queryN format
            parts = re.split(r"(?im)^--\s*query[^\n]*\n", text)

            # parts[0] — ignore preamble
            for block in parts[1:]:
                q = block.strip()
                if q:
                    queries.append(q)

        return queries

    lines = [
        line.strip()
        for line in f.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return lines

