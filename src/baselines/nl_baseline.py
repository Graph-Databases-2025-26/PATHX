from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

from ..utils.dataset_io import discover_datasets, read_nl_queries, dataset_dir
from ..config.config import load_config, AppConfig
from .llm_factory import get_llm_wrapper
from .core.chains import build_lcel_chain, build_followup_chain
from .core.iterative_runner import run_iterative_baseline
from .core.output_io import save_query_result_json


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def run_nl_baseline_for_dataset(
    config: AppConfig,
    dataset: str,
    strategy: str,
    max_iters: int = 5,
) -> None:
    """
    Run the NL baseline for a single dataset using the selected LLM provider
    and prompting strategy (direct / zero_shot / few_shot).

    Pipeline:
      - load NL questions from data/<DATASET>/queries_*dataset_name*.txt
      - build LCEL chain (context + Jinja2 prompt + LLM)
      - perform iterative querying (initial + follow-up "more results")
      - save each query result to JSON
    """

    ds_dir: Path = dataset_dir(dataset)
    if not ds_dir.exists():
        logger.error(f"[NL baseline] Dataset directory does not exist: {ds_dir}")
        return

    # 1) Load natural-language questions
    try:
        nl_queries: List[str] = read_nl_queries(ds_dir)
    except FileNotFoundError as e:
        logger.warning(f"[NL baseline] No nl_queries.txt found for {dataset}: {e}")
        return

    if not nl_queries:
        logger.warning(f"[NL baseline] No NL queries loaded for dataset {dataset}")
        return

    # 2) Init LLM wrapper and LCEL chains
    llm_wrapper = get_llm_wrapper(config)
    provider_name = llm_wrapper.get_provider_name()

    logger.info(
        f"[NL baseline] Dataset={dataset} | provider={provider_name} | strategy={strategy} "
        f"| queries={len(nl_queries)} | max_iters={max_iters}"
    )

    initial_chain = build_lcel_chain(llm_wrapper, baseline_type="NL")
    followup_chain = build_followup_chain(llm_wrapper)

    # 3) Run baseline query-by-query
    for idx, question in enumerate(nl_queries, start=1):
        logger.info(f"[NL baseline] {dataset} Q{idx}/{len(nl_queries)}")

        # Payload for context_builder (get_db_context)
        payload = {
            "database": dataset.lower(),
            "b_type": "NL",
            "sql": None,
            "nl_question": question,
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
                f"[NL baseline] {dataset} Q{idx} – LLM/iterative runner failed: {e}"
            )
            error_payload = {
                "_error": str(e),
                "result_set": [],
                "time": None,
                "tokens": None,
            }
            save_query_result_json(
                baseline_type="NL",
                dataset=dataset,
                provider=provider_name,
                query_idx=idx,
                data=error_payload,
            )
            continue

        # 4) Save JSON result:
        #    { "result_set": [...], "time": float|None, "tokens": Any }
        save_query_result_json(
            baseline_type="NL",
            dataset=dataset,
            provider=provider_name,
            query_idx=idx,
            data=result,
        )

        logger.info(f"[NL baseline] {dataset} Q{idx} – saved.")


def main() -> None:
    """
    CLI entrypoint for NL baseline.

    Examples:
        python -m src.baselines.nl_baseline --strategy direct
        python -m src.baselines.nl_baseline --datasets flight-2 imdb --strategy few_shot
    """
    ap = argparse.ArgumentParser(
        description="Run NL baseline using LCEL, Jinja2 prompts and iterative querying."
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
        help="Prompting strategy to use for NL baseline.",
    )
    ap.add_argument(
        "--max-iters",
        type=int,
        default=5,
        help="Maximum number of follow-up iterations per NL query.",
    )

    args = ap.parse_args()

    config = load_config()

    datasets = args.datasets or discover_datasets()
    if not datasets:
        logger.error("No datasets found under ./data")
        return

    logger.info(
        f"[NL baseline] Running for datasets: {', '.join(datasets)} "
        f"| strategy={args.strategy} | max_iters={args.max_iters}"
    )

    for ds in datasets:
        run_nl_baseline_for_dataset(
            config=config,
            dataset=ds,
            strategy=args.strategy,
            max_iters=args.max_iters,
        )


if __name__ == "__main__":
    main()
