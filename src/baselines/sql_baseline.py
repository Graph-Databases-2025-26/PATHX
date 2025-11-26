from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..utils.dataset_io import discover_datasets, read_queries
from .llm_factory import get_llm_wrapper
from .core.chains import build_lcel_chain, build_followup_chain
from .core.iterative_runner import run_iterative_baseline
from .core.output_io import save_query_result_json

from config import load_config, AppConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def run_sql_baseline_for_dataset(
    config: AppConfig,
    dataset: str,
    strategy: str,
    max_iters: int = 5,
) -> None:
    """
    Run the SQL baseline for a single dataset using the selected LLM provider
    and prompting strategy (direct / zero_shot / few_shot).

    Pipeline:
      - load SQL queries from data/<DATASET>/queries_*.sql
      - build LCEL chain (context + Jinja2 prompt + LLM)
      - perform iterative querying (initial + follow-up "more results")
      - save each query result to JSON
    """
    llm = get_llm_wrapper(config)
    provider_name = llm.get_provider_name()

    initial_chain = build_lcel_chain(llm, baseline_type="SQL")
    followup_chain = build_followup_chain(llm)

    ds_dir = Path("data") / dataset
    if not ds_dir.exists():
        logger.error(f"[SQL baseline] Dataset directory does not exist: {ds_dir}")
        return

    queries = read_queries(ds_dir)
    if not queries:
        logger.warning(f"[SQL baseline] No queries_*.sql found for dataset {dataset}")
        return

    logger.info(
        f"[SQL baseline] Dataset={dataset} | provider={provider_name} | strategy={strategy} "
        f"| queries={len(queries)} | max_iters={max_iters}"
    )

    for idx, sql in enumerate(queries, start=1):
        logger.info(f"[SQL baseline] {dataset} Q{idx}/{len(queries)}")

        payload = {
            "database": dataset.lower(),
            "b_type": "SQL",
            "sql": sql,
            "nl_question": None,
            "strategy": strategy,
        }

        try:
            result = run_iterative_baseline(
                initial_chain=initial_chain,
                followup_chain=followup_chain,
                initial_payload=payload,
                max_iters=max_iters,
            )
        except Exception as e:
            logger.error(
                f"[SQL baseline] {dataset} Q{idx} – LLM/iterative runner failed: {e}"
            )
            error_payload = {
                "_error": str(e),
                "result_set": [],
                "time": None,
                "tokens": None,
            }
            save_query_result_json(
                baseline_type="SQL",
                dataset=dataset,
                provider=provider_name,
                query_idx=idx,
                data=error_payload,
            )
            continue

        save_query_result_json(
            baseline_type="SQL",
            dataset=dataset,
            provider=provider_name,
            query_idx=idx,
            data=result,
        )
        logger.info(f"[SQL baseline] {dataset} Q{idx} – saved.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run SQL baseline using LCEL, Jinja2 prompts and iterative querying."
    )
    ap.add_argument(
        "--datasets",
        nargs="*",
        help=(
            "Optional list of dataset folder names under ./data. "
            "If omitted, all datasets discovered in ./data are used."
        ),
    )
    ap.add_argument(
        "--strategy",
        choices=["direct", "zero_shot", "few_shot"],
        default="direct",
        help="Prompting strategy to use for SQL baseline.",
    )
    ap.add_argument(
        "--max-iters",
        type=int,
        default=5,
        help="Maximum number of follow-up iterations per SQL query.",
    )

    args = ap.parse_args()

    config = load_config()

    datasets = args.datasets or discover_datasets()
    if not datasets:
        logger.error("No datasets found under ./data")
        return

    logger.info(
        f"[SQL baseline] Running for datasets: {', '.join(datasets)} "
        f"| strategy={args.strategy} | max_iters={args.max_iters}"
    )

    for ds in datasets:
        run_sql_baseline_for_dataset(
            config=config,
            dataset=ds,
            strategy=args.strategy,
            max_iters=args.max_iters,
        )


if __name__ == "__main__":
    main()
