from __future__ import annotations
from pathlib import Path
import re

DATA = Path("data")

def discover_datasets() -> list[str]:
    """
    Discovers and returns a list of dataset directory names in the DATA directory.
    """
    out: list[str] = []
    for p in sorted(DATA.iterdir()):
        if p.is_dir() and not p.name.startswith("."):
            out.append(p.name)
    return out

def read_queries(ds_dir: Path) -> list[str]:
    """
    Reads and returns SQL queries from the specified dataset directory.
    """
    qfiles = sorted(ds_dir.glob("queries*.sql")) or sorted(ds_dir.glob("queries_*.sql"))
    if not qfiles:
        raise FileNotFoundError(f"No queries_*.sql found in {ds_dir}")
    text = qfiles[0].read_text(encoding="utf-8")
    # remove line comments
    lines = [ln for ln in text.splitlines() if not ln.strip().startswith("--")]
    text = "\n".join(lines)
    # split by semicolon (robust)
    parts = [p.strip() for p in re.split(r';\s*(?:--.*)?\n', text) if p.strip()]
    # second pass + de-dup
    stmts, seen = [], set()
    for p in parts:
        for s in [x.strip() for x in p.split(";") if x.strip()]:
            if s and s not in seen:
                seen.add(s)
                stmts.append(s)
    return stmts

def read_nl_queries(ds_dir: Path) -> list[str]:
    """
    Reads and returns natural language queries from the specified dataset directory.
    """
    qfiles = sorted(ds_dir.glob("queries*.txt")) or sorted(ds_dir.glob("queries_*.txt"))
    if not qfiles:
        raise FileNotFoundError(f"No queries_*.txt found in {ds_dir}")
    text = qfiles[0].read_text(encoding="utf-8")
    queries = [
        ln.strip()
        for ln in text.splitlines()
        if ln.strip() and not ln.strip().startswith("--")
    ]
    return queries