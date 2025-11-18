from __future__ import annotations
from pathlib import Path
import re

DATA = Path("data")

def discover_datasets() -> list[str]:
    out: list[str] = []
    for p in sorted(DATA.iterdir()):
        if p.is_dir() and not p.name.startswith("."):
            out.append(p.name)
    return out

def read_queries(ds_dir: Path) -> list[str]:
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
