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

def build_prompt_strict_json(dataset: str, sql: str) -> str:
    return f"""
You are a deterministic SQL execution engine.

You will execute the following SQL query on the "{dataset}" database.
Return ONLY the execution result as a **valid JSON array**.

STRICT RULES:
- Output MUST be valid JSON.
- Top-level MUST be a JSON list.
- Each row MUST be a JSON object {{column: value}}.
- NO explanations.
- NO Markdown.
- NO comments.
- NO surrounding text.
- DO NOT include keys like "result", "data", "rows".

SQL query:
{sql}
""".strip()

def build_prompt_json_example(dataset: str, sql: str) -> str:
    return f"""
You are an expert SQL interpreter.

Execute the SQL query on the "{dataset}" database and return ONLY the result rows.

Your response MUST follow this JSON template:

[
  {{"column1": "value1", "column2": 123}},
  {{"column1": "value2", "column2": 456}}
]

Your output MUST:
- be valid JSON
- contain one object per row
- use correct column names
- contain no text before or after the JSON

SQL query:
{sql}
""".strip()

def build_prompt_constrained(dataset: str, sql: str) -> str:
    return f"""
You simulate a SQL execution engine.

TASK:
Return ONLY the SQL query result as a JSON ARRAY.

HARD REQUIREMENTS:
1. Output must be **valid JSON**.
2. Top-level must be a **list**.
3. Each row must be an **object**.
4. Keys must be **exact SQL column names**.
5. Do NOT describe, explain, comment, or format with markdown.
6. Do NOT include text outside the JSON.

FAILURE CONDITIONS (do NOT do these):
- No "Here is the result".
- No "```json".
- No extra keys like "output" or "result".

Now execute:

{sql}
""".strip()

def build_prompt_hidden_cot(dataset: str, sql: str) -> str:
    return f"""
You are an advanced SQL interpreter.

You may perform internal reasoning to understand the query,
but the FINAL OUTPUT must be ONLY a JSON array of rows.

INVISIBLE RULES:
- Think step-by-step internally, but DO NOT reveal the reasoning.
- Return ONLY the final JSON array.
- Must be valid JSON.
- No markdown, comments, or text outside JSON.

SQL query:
{sql}
""".strip()

def build_prompt_fewshot(dataset: str, sql: str) -> str:
    return f"""
You are a SQL execution engine.

BAD EXAMPLE (DO NOT DO THIS):
Result:
[
  {{ "value": 1 }}
]

GOOD EXAMPLE (DO THIS):
[
  {{"columnA": "abc", "columnB": 123}},
  {{"columnA": "def", "columnB": 456}}
]

Follow these rules:
- Valid JSON only.
- JSON array only.
- Each row is an object with correct column names.
- No explanation, no markdown.

Execute on database "{dataset}" this SQL:

{sql}
""".strip()

def build_prompt_compact(dataset: str, sql: str) -> str:
    return f"""
Execute SQL on database "{dataset}" and output ONLY valid JSON:

RULE:
[
  {{col1: val1, col2: val2}},
  ...
]

NOTHING before or after the JSON.

SQL:
{sql}
""".strip()

def build_prompt(strategy: str, dataset: str, sql: str) -> str:
    strategies = {
        "strict_json": build_prompt_strict_json,
        "json_example": build_prompt_json_example,
        "constrained": build_prompt_constrained,
        "hidden_cot": build_prompt_hidden_cot,
        "fewshot": build_prompt_fewshot,
        "compact": build_prompt_compact,
    }
    if strategy not in strategies:
        raise ValueError(f"Unknown strategy: {strategy}")
    return strategies[strategy](dataset, sql)



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

def run_sql_baseline_for_dataset(dataset: str, strategy: str) -> None:
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

    logger.info(f"[SQL baseline] Dataset: {dataset} ({len(sqls)} queries), strategy={strategy}")

    for i, sql in enumerate(sqls, start=1):
        logger.info(f"[{dataset}] Q{i}/{len(sqls)}")
        prompt = build_prompt(strategy, dataset, sql)   # ⬅️ use strategy here

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
        # run for all datasets discovered in ./data with default strategy
        python -m src.pipelines.sql_baseline

        # run only for movies and geo with a specific strategy
        python -m src.pipelines.sql_baseline --datasets movies geo --strategy strict_json
    """
    ap = argparse.ArgumentParser(description="Run SQL baseline (Baseline 3) with Watsonx LLM.")
    ap.add_argument(
        "--datasets",
        nargs="*",
        help="Optional list of dataset names (as folder names under ./data). "
             "If omitted, all datasets discovered in ./data are used.",
    )
    ap.add_argument(
        "--strategy",
        choices=["strict_json", "json_example", "constrained", "hidden_cot", "fewshot", "compact"],
        default="strict_json",
        help="Prompting strategy to use for the SQL baseline.",
    )

    args = ap.parse_args()

    datasets = args.datasets or discover_datasets()
    logger.info(
        f"Running SQL baseline for datasets: {', '.join(datasets)} "
        f"with strategy={args.strategy}"
    )

    for ds in datasets:
        run_sql_baseline_for_dataset(ds, args.strategy)


if __name__ == "__main__":
    main()
