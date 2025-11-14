# src/pipelines/sql_baseline.py
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import List, Any

from ..llm.watsonx_client import generate_text
from .common import discover_datasets, read_queries

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DATA_ROOT = Path("data")
OUT_ROOT = Path("sql_baseline_results")

def _build_prompt(dataset: str, sql: str) -> str:
    """
    Build the prompt for the SQL baseline (Baseline 3 in the slides).

    The LLM receives a SQL query and must return ONLY a JSON array of rows,
    each row as an object {column: value}.
    """
    return f"""
You are a SQL execution engine.

You receive a SQL query that is executed on a relational database called "{dataset}".
Your task is to RETURN ONLY THE RESULT SET as a JSON array.

Strict rules:
- Output MUST be valid JSON.
- The top-level structure MUST be a list.
- Each row MUST be a JSON object with column names as keys.
- Do NOT include explanations, comments, or markdown formatting.
- Do NOT include any keys like "result", "data", "rows" – just the raw array.

Example of the REQUIRED format:

[
  {{"column1": "value1", "column2": 123}},
  {{"column1": "value2", "column2": 456}}
]

Now answer for this query:

{sql}
""".strip()


def _try_extract_json(text: str) -> Any:
    """
    Try to parse JSON from the LLM output.
    1) direct json.loads
    2) heuristic: first '[' ... last ']'
    Fallback: return dict with error + raw text.
    """
    text = text.strip()

    # 1) direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # 2) extract between first '[' and last ']'
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            # continue to fallback
            pass

    # 3) fallback object (won't evaluate well, but at least is JSON)
    logger.warning("Failed to parse LLM output as JSON, storing raw text.")
    return {
        "_error": "failed_to_parse_json",
        "raw_text": text,
    }


def _dataset_dir(name: str) -> Path:
    return DATA_ROOT / name


# ---------- core logic ----------

def run_sql_baseline_for_dataset(dataset: str) -> None:
    """
    For a single dataset:
    - read queries_*.sql
    - send each SQL query to the LLM
    - save results to sql_baseline_results/<DATASET>/query<i>.json
    """
    ds_dir = _dataset_dir(dataset)
    if not ds_dir.exists():
        logger.error(f"Dataset directory not found: {ds_dir}")
        return

    out_dir = OUT_ROOT / dataset.upper()
    out_dir.mkdir(parents=True, exist_ok=True)

    sqls: List[str] = read_queries(ds_dir)
    if not sqls:
        logger.warning(f"No queries_*.sql found for dataset {dataset}")
        return

    logger.info(f"[SQL baseline] Dataset: {dataset} ({len(sqls)} queries)")

    for i, sql in enumerate(sqls, start=1):
        logger.info(f"[{dataset}] Q{i}/{len(sqls)}")
        prompt = _build_prompt(dataset, sql)

        try:
            raw_answer = generate_text(prompt)
        except Exception as e:
            logger.error(f"[{dataset}] Q{i} – LLM call failed: {e}")
            data = {"_error": str(e), "prompt": prompt}
        else:
            data = _try_extract_json(raw_answer)

        out_path = out_dir / f"query{i}.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"[{dataset}] Q{i} → {out_path}")


def main() -> None:
    """
    CLI entry point.

    Examples:
        # run for all datasets discovered in ./data
        python -m src.pipelines.sql_baseline

        # run only for movies and geo
        python -m src.pipelines.sql_baseline --datasets movies geo
    """
    ap = argparse.ArgumentParser(description="Run SQL baseline (Baseline 3) with Watsonx LLM.")
    ap.add_argument(
        "--datasets",
        nargs="*",
        help="Optional list of dataset names (as folder names under ./data). "
             "If omitted, all datasets discovered in ./data are used.",
    )
    args = ap.parse_args()

    datasets = args.datasets or discover_datasets()
    logger.info(f"Running SQL baseline for datasets: {', '.join(datasets)}")

    for ds in datasets:
        run_sql_baseline_for_dataset(ds)


if __name__ == "__main__":
    main()
