from __future__ import annotations
from pathlib import Path
from typing import Dict
import json
import time
import duckdb

from src.utils.logging import logger, log_query_event

RESULTS_ROOT = Path("results")
JSON_ROOT = RESULTS_ROOT / "explain_json"
TXT_ROOT  = RESULTS_ROOT / "explain_txt"


def _ensure_db(db_path: Path | str) -> Path:
    p = Path(db_path)
    if not p.exists():
        logger.warning(f"Database file not found: {p}")
        raise FileNotFoundError(f"Database file not found: {p}")
    return p


def _run_explain_text(db_path: Path | str, sql: str, analyze: bool) -> str:
    db = _ensure_db(db_path)
    con = duckdb.connect(str(db), read_only=True)
    try:
        stmt = f"EXPLAIN {'ANALYZE ' if analyze else ''}{sql}"
        rows = con.execute(stmt).fetchall()  # list[tuple[str, ...]]
        # Join tuple columns per row; then join rows by newline
        plan_lines = [" ".join(str(c) for c in row if c is not None) for row in rows]
        return "\n".join(plan_lines)
    finally:
        con.close()


def _clean_plan_text(text: str) -> str:
    norm = text.replace("\r\n", "\n").replace("\r", "\n").lstrip()
    clean_lines: list[str] = []
    for line in norm.splitlines():
        ls = line.strip().lower()
        # Remove top banner lines DuckDB prints in some versions
        if ls.startswith("analyzed_plan"):
            continue
        if ls.startswith("physical_plan"):
            continue
        if ls.startswith("explain analyze") or ls.startswith("explain "):
            continue
        clean_lines.append(line)
    return ("\n".join(clean_lines).rstrip() + "\n") if clean_lines else norm


def _dataset_from_db(db_path: Path | str | None) -> str:
    if db_path:
        return Path(db_path).stem.upper()
    return "UNKNOWN"


def _write_json(plan: Dict[str, str], dataset: str, filename: str) -> Path:
    out_dir = JSON_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Saved JSON plan → {out_path}")
    return out_path


def _write_txt(text: str, dataset: str, filename: str) -> Path:
    out_dir = TXT_ROOT / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / filename
    out_path.write_text(text, encoding="utf-8")
    logger.info(f"Saved TXT plan  → {out_path}")
    return out_path


def get_explain_text(db_path: Path | str, sql: str) -> Dict[str, str]:
    log_query_event("EXPLAIN", sql)
    start = time.time()
    plan_text = _run_explain_text(db_path, sql, analyze=False)
    elapsed = (time.time() - start) * 1000.0
    logger.info(f"EXPLAIN completed latency_ms={elapsed:.1f}")
    return {"format": "text", "plan_text": plan_text}


def get_explain_analyze_text(db_path: Path | str, sql: str) -> Dict[str, str]:
    log_query_event("EXPLAIN_ANALYZE", sql)
    start = time.time()
    plan_text = _run_explain_text(db_path, sql, analyze=True)
    elapsed = (time.time() - start) * 1000.0
    logger.info(f"EXPLAIN ANALYZE completed latency_ms={elapsed:.1f}")
    return {"format": "text", "plan_text": plan_text}


def save_text_plan(
    plan: Dict[str, str],
    out_json: Path | str,
    out_txt: Path | str | None = None,
    *,
    db_path: Path | str | None = None,
) -> Path:
    out_json = Path(out_json)
    dataset = _dataset_from_db(db_path)

    # JSON
    json_filename = out_json.name
    json_path = _write_json(plan, dataset, json_filename)

    # TXT
    if out_txt:
        txt_filename = Path(out_txt).name
        clean_text = _clean_plan_text(plan.get("plan_text", ""))
        _write_txt(clean_text, dataset, txt_filename)

    return json_path


def get_explain(db_path: Path | str, sql: str):
    return get_explain_text(db_path, sql)


def get_explain_analyze(db_path: Path | str, sql: str):
    return get_explain_analyze_text(db_path, sql)


def save_explain(db_path: Path | str, sql: str, out_path: Path | str):
    plan = get_explain_text(db_path, sql)
    return save_text_plan(plan, out_path, db_path=db_path)


def save_analyze(db_path: Path | str, sql: str, out_path: Path | str):
    plan = get_explain_analyze_text(db_path, sql)
    return save_text_plan(plan, out_path, db_path=db_path)


def save_explain_pair(db_path: Path | str, sql: str, out_base: Path | str):
    base = Path(out_base)
    p_explain = base.with_name(base.name + "__explain.json")
    p_analyze = base.with_name(base.name + "__analyze.json")
    j1 = save_explain(db_path, sql, p_explain)
    j2 = save_analyze(db_path, sql, p_analyze)
    return j1, j2


def save_both(db_path: Path | str, sql: str, out_base: Path | str):
    base = Path(out_base)

    # EXPLAIN command
    plan = get_explain(db_path, sql)
    json_explain = base.with_name(base.name + "__explain.json")
    txt_explain  = base.with_name(base.name + "__explain.txt")
    j1 = save_text_plan(plan, json_explain, txt_explain, db_path=db_path)

    # EXPLAIN ANALYZE command
    plan_an = get_explain_analyze(db_path, sql)
    json_analyze = base.with_name(base.name + "__analyze.json")
    txt_analyze  = base.with_name(base.name + "__analyze.txt")
    j2 = save_text_plan(plan_an, json_analyze, txt_analyze, db_path=db_path)

    return j1, (TXT_ROOT / _dataset_from_db(db_path) / txt_explain.name), \
           j2, (TXT_ROOT / _dataset_from_db(db_path) / txt_analyze.name)
