from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference
import google.generativeai as genai
import json
import os
import logging
import re
from typing import List, Dict, Any, Callable
import time
from pathlib import Path
from importlib import import_module
import argparse

from .common import discover_datasets, read_queries

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


DATA_ROOT = Path("data")
OUT_ROOT = Path("nl_baseline_results")
PROVIDER_CHOICES = ("azure","gemini", "watsonx")

def _load_provider(name: str) -> Callable[[str], Any]:
    
    try:
        if name == "azure":
            return import_module("src.llm.azure_client").generate_text
        if name == "gemini":
            return import_module("src.llm.gemini_client").generate_text
        if name == "watsonx":
            return import_module("src.llm.watsonx_client").generate_text
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM provider '{name}': {e}")
    raise ValueError(f"Unknown provider: {name}")

def _build_prompt(nl_query: str) -> str:
    return f"""
    Answer the following query by giving a JSON array.
    Make sure the output is valid JSON and uses consistent object keys.
    No additional text is needed.

    Query:
    {nl_query}

    """

def _try_extract_json(text: str) -> List[Dict]:
    
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
    return [
        {"_error": "failed_to_parse_json",
        "raw_text": text,}
            ]


def _dataset_dir(name: str) -> Path:
    return DATA_ROOT / name


def run_nl_baseline_for_dataset(dataset: str, llm_fn: Callable[[str], Any]) -> None:
    ds_dir = _dataset_dir(dataset)
    if not ds_dir.exists():
        logger.error(f"Dataset directory not found: {ds_dir}")
        return

    out_dir = OUT_ROOT / dataset.upper()
    out_dir.mkdir(parents=True, exist_ok=True)

    queries: List[str] = read_queries(ds_dir)
    if not queries:
        logger.warning(f"No queries_*.sql found for dataset {dataset}")
        return

    logger.info(f"[NL baseline] Dataset: {dataset} ({len(queries)} queries)")

    for i, query in enumerate(queries, start=1):
        logger.info(f"[{dataset}] Q{i}/{len(queries)}")
        prompt = _build_prompt(query)

        elapsed = None 

        tokens = None
        try:
            t0 = time.time()
            result = llm_fn(prompt)
            elapsed = time.time() - t0
            if isinstance(result, (tuple, list)):
                raw_answer = result[0]
                tokens = result[1] if len(result) > 1 else None
            else:
                raw_answer = result
        except Exception as e:
            logger.error(f"[{dataset}] Q{i} – LLM call failed: {e}")
            data = {"_error": str(e), "prompt": prompt}
        else:
            data = _try_extract_json(raw_answer)

        # Full JSON wrapping
        if isinstance(data, list):
            data = {"result_set": data}
        elif isinstance(data, dict) and "result_set" not in data:
            data = {"result_set": [data]}

        # Add time/tokens
        if isinstance(data, dict):
            data.setdefault("tokens", tokens)
            # If LLM called successfully — write time, else None
            data["time"] = round(elapsed, 3) if elapsed is not None else None

        out_path = out_dir / f"query{i}.json"
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info(f"[{dataset}] Q{i} → {out_path}")

def main() -> None:
    ap = argparse.ArgumentParser(description="Run NL baseline (Baseline 2) with select LLM backend.")
    ap.add_argument(
        "--datasets",
        nargs="*",
        help="Optional list of dataset names (as folder names under ./data). "
             "If omitted, all datasets discovered in ./data are used.",
    )
    ap.add_argument(
        "--provider",
        choices=PROVIDER_CHOICES,
        default=os.getenv("LLM_PROVIDER", "watsonx"),
        help="LLM backend to use (default from LLM_PROVIDER env, falls back to watsonx).",
    )

    args = ap.parse_args()

    datasets = args.datasets or discover_datasets()
    llm_fn = _load_provider(args.provider)
    logger.info(
        f"Running SQL baseline for datasets: {', '.join(datasets)} "
        f"with provider={args.provider}"
    )

    for ds in datasets:
        run_nl_baseline_for_dataset(ds, llm_fn)


if __name__ == "__main__":
    main()