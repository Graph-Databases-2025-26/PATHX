from __future__ import annotations

import argparse
import json
import logging
import os
import time
from importlib import import_module
from pathlib import Path
from typing import List, Dict, Any, Callable

from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from .common import discover_datasets, read_nl_queries

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

DATA_ROOT = Path("data")
OUT_ROOT = Path("nl_baseline_results")
PROVIDER_CHOICES = ("azure", "gemini", "watsonx")


class Response(BaseModel):
    result_set: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of result records, each as a {column_name: value} dict.",
    )

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


def _strip_markdown_fences(text: str) -> str:
    if text is None:
        return ""
    text = text.strip()
    if text.startswith("```"):
        first_newline = text.find("\n")
        if first_newline != -1:
            text = text[first_newline + 1 :]
        last_fence = text.rfind("```")
        if last_fence != -1:
            text = text[:last_fence]
    return text.strip()


def _build_prompt(nl_query: str) -> str:
    prompt = f"""
    Answer the following query by giving a JSON array.

    Query:
    {nl_query}

    """.strip()

    return prompt


def _try_extract_json(text: str) -> List[Dict[str, Any]]:
    text = text.strip()

    # 1) direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
        return [parsed]
    except Exception:
        pass

    # 2) extract between first '[' and last ']'
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, list):
                return parsed
            return [parsed]
        except Exception:
            pass

    # 3) fallback object
    logger.warning("Failed to parse LLM output as JSON, storing raw text.")
    return [
        {
            "_error": "failed_to_parse_json",
            "raw_text": text,
        }
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

    queries: List[str] = read_nl_queries(ds_dir)
    if not queries:
        logger.warning(f"No queries_*.sql (NL files) found for dataset {dataset}")
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
            data: Dict[str, Any] = {"_error": str(e), "prompt": prompt}
        else:
            try:
                parsed: Response = PARSER.parse(
                    _strip_markdown_fences(str(raw_answer))
                )
                data = {"result_set": parsed.result_set}
            except Exception as e:
                logger.warning(
                    f"[{dataset}] Q{i} – Pydantic parsing failed "
                    f"({type(e).__name__}), falling back to simple JSON extraction."
                )
                rs = _try_extract_json(str(raw_answer))
                data = {"result_set": rs}

        if isinstance(data, dict):
            data.setdefault("tokens", tokens)
            data["time"] = round(elapsed, 3) if elapsed is not None else None

        out_path = out_dir / f"query{i}.json"
        out_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"[{dataset}] Q{i} → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run NL baseline (Baseline 2) with select LLM backend."
    )
    ap.add_argument(
        "--datasets",
        nargs="*",
        help=(
            "Optional list of dataset names (as folder names under ./data). "
            "If omitted, all datasets discovered in ./data are used."
        ),
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
        f"Running NL baseline for datasets: {', '.join(datasets)} "
        f"with provider={args.provider}"
    )

    for ds in datasets:
        run_nl_baseline_for_dataset(ds, llm_fn)


if __name__ == "__main__":
    main()
