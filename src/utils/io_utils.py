from pathlib import Path
import json
import re

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def read_sql_statements(path: Path) -> list[str]:
    """Robustly split queries_*.sql into a list of SQL statements."""
    text = path.read_text(encoding="utf-8")
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("--"):
            continue
        lines.append(line)
    text = "\n".join(lines)
    parts = [p.strip() for p in re.split(r';\s*(?:--.*)?\n', text) if p.strip()]
    stmts = []
    for p in parts:
        for q in [s.strip() for s in p.split(';') if s.strip()]:
            if q:
                stmts.append(q)
    # de-dup while preserving order
    out, seen = [], set()
    for s in stmts:
        if s not in seen:
            seen.add(s); out.append(s)
    return out
