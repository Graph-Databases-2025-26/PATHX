"""
Filter-LLMScan baseline runner.

Executes queries using Filter-LLMScan operator with different pushdown strategies
and saves results for analysis.
"""

from __future__ import annotations

import argparse
import logging
import json
from pathlib import Path
from typing import Optional

# Assurez-vous que ces imports correspondent à votre structure de projet
from src.utils.dataset_io import discover_datasets, read_queries
from src.config.config import load_config, AppConfig
from src.baselines.llm_factory import get_llm_wrapper
# Import de la classe et de l'Enum à partir du fichier que vous avez fourni
from src.galois.filter_llm_scan import FilterLLMScanOperator, PushdownStrategy
from src.baselines.core.output_io import save_query_result_json


logger = logging.getLogger(__name__)
# On s'assure que le niveau de log est au moins INFO pour voir les résultats
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


def parse_sql_query(sql: str) -> dict:
    """
    Parse SQL query to extract components for Filter-LLMScan.
    
    Simple parser that extracts:
    - SELECT attributes
    - FROM table
    - WHERE clause
    """
    sql = sql.strip()
    sql_upper = sql.upper()
    
    result = {
        "table": None,
        "attributes": [],
        "where_clause": None,
    }
    
    # Extract SELECT attributes
    if "SELECT" in sql_upper:
        select_start = sql_upper.index("SELECT") + 6
        from_pos = sql_upper.index("FROM") if "FROM" in sql_upper else len(sql)
        select_part = sql[select_start:from_pos].strip()
        
        if select_part.upper() != "*":
            # Parse comma-separated attributes
            attrs = [attr.strip() for attr in select_part.split(",")]
            result["attributes"] = attrs
    
    # Extract FROM table
    if "FROM" in sql_upper:
        from_start = sql_upper.index("FROM") + 4
        where_pos = sql_upper.index("WHERE") if "WHERE" in sql_upper else len(sql)
        table_part = sql[from_start:where_pos].strip()
        # Remove any JOIN or other clauses (simple parsing)
        result["table"] = table_part.split()[0].strip()
    
    # Extract WHERE clause
    if "WHERE" in sql_upper:
        where_start = sql_upper.index("WHERE") + 5
        where_part = sql[where_start:].strip()
        # Remove ORDER BY, LIMIT, etc. (simple parsing)
        for keyword in ["ORDER BY", "GROUP BY", "LIMIT", "HAVING"]:
            if keyword in where_part.upper():
                where_part = where_part[:where_part.upper().index(keyword)].strip()
        result["where_clause"] = where_part
    
    return result


def run_filter_llmscan_for_dataset(
    config: AppConfig,
    dataset: str,
    strategy: PushdownStrategy,
    provider_name: str,
    max_iters: int = 5,
) -> None:
    """
    Run Filter-LLMScan baseline for a single dataset.
    """
    llm_wrapper = get_llm_wrapper(config)
    llm = llm_wrapper.get_llm_instance()
    
    # Create Filter-LLMScan operator
    operator = FilterLLMScanOperator(
        llm=llm,
        max_iters=max_iters,
        pushdown_strategy=strategy
    )
    
    ds_dir = Path("data") / dataset
    if not ds_dir.exists():
        logger.error(f"[Filter-LLMScan] Dataset directory does not exist: {ds_dir}")
        return
    
    queries = read_queries(ds_dir)
    if not queries:
        logger.warning(f"[Filter-LLMScan] No queries_*.sql found for dataset {dataset}")
        return
    
    logger.info(
        f"[Filter-LLMScan] Dataset={dataset} | provider={provider_name} | "
        f"strategy={strategy.value} | queries={len(queries)} | max_iters={max_iters}"
    )
    
    for idx, sql in enumerate(queries, start=1):
        logger.info(f"[Filter-LLMScan] {dataset} Q{idx}/{len(queries)}")
        logger.debug(f"SQL: {sql}")
        
        try:
            # Parse SQL query
            parsed = parse_sql_query(sql)
            
            if not parsed["table"]:
                raise ValueError(f"Could not parse table from SQL: {sql}")
            
            logger.debug(
                f"Parsed: table={parsed['table']}, "
                f"attributes={parsed['attributes']}, "
                f"where={parsed['where_clause']}"
            )
            
            # Execute Filter-LLMScan
            result = operator.apply_pushdown(
                dataset=dataset.upper(),
                table=parsed["table"],
                attributes=parsed["attributes"] if parsed["attributes"] else None,
                where_clause=parsed["where_clause"],
            )
            
            # Add SQL query to result for reference
            result["sql"] = sql
            result["parsed_query"] = parsed
            
            logger.info(
                f"[Filter-LLMScan] {dataset} Q{idx} — "
                f"Results: {len(result['result_set'])}, "
                f"Time: {result['time']}s, "
                f"Tokens: {result['tokens']}"
            )
            
            # Log pushdown decision
            push_meta = result.get("meta", {})
            if "pushdown_strategy" in push_meta:
                pushed = push_meta.get("pushed_predicates", 0)
                post = push_meta.get("post_filter_predicates", 0)
                logger.info(f"  Strategy: {push_meta['pushdown_strategy']} | Pushed: {pushed}, Post-Filter: {post}")
            
        except Exception as e:
            logger.error(
                f"[Filter-LLMScan] {dataset} Q{idx} — Failed: {e}",
                exc_info=True
            )
            result = {
                "_error": str(e),
                "sql": sql,
                "result_set": [],
                "time": None,
                "tokens": None,
                "meta": {}
            }
        
        # Save result with new folder structure
        save_filter_llmscan_result(
            provider=provider_name,
            strategy=strategy.value,
            dataset=dataset,
            query_idx=idx,
            data=result,
        )
        logger.info(f"[Filter-LLMScan] {dataset} Q{idx} — saved.")


def save_filter_llmscan_result(
    provider: str,
    strategy: str,
    dataset: str,
    query_idx: int,
    data: dict,
) -> None:
    """
    Save Filter-LLMScan query result to JSON file.
    
    Creates directory structure:
    filter_llm_scan_results/{provider}/{strategy}/{dataset}/query_{idx}.json
    """
    # MODIFICATION: Structure changée de dataset/provider/strategy à provider/strategy/dataset
    results_dir = Path("filter_llm_scan_results") / provider / strategy / dataset
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = results_dir / f"query{query_idx}.json"
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    logger.debug(f"Saved result to {output_file}")


def main() -> None:
    """Main entry point for Filter-LLMScan baseline."""
    ap = argparse.ArgumentParser(
        description="Run Filter-LLMScan baseline with different pushdown strategies."
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
        choices=[e.value for e in PushdownStrategy],
        default=PushdownStrategy.ALL.value,
        help="Pushdown strategy to use (choices: none, single, all).",
    )
    ap.add_argument(
        "--max-iters",
        type=int,
        default=5,
        help="Maximum number of follow-up iterations per query (default: 5).",
    )
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    
    args = ap.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Load configuration
    config = load_config()
    
    # Map string to enum
    strategy = PushdownStrategy(args.strategy)
    
    # Get provider name once
    llm_wrapper = get_llm_wrapper(config)
    provider_name = llm_wrapper.get_provider_name()
    
    # Discover datasets
    datasets = args.datasets or discover_datasets()
    if not datasets:
        logger.error("No datasets found under ./data")
        return
    
    logger.info(
        f"[Filter-LLMScan] Running for datasets: {', '.join(datasets)} "
        f"| provider={provider_name} | strategy={args.strategy} | max_iters={args.max_iters}"
    )
    
    for ds in datasets:
        run_filter_llmscan_for_dataset(
            config=config,
            dataset=ds,
            strategy=strategy,
            provider_name=provider_name,
            max_iters=args.max_iters,
        )
    
    logger.info("[Filter-LLMScan] All datasets completed.")


if __name__ == "__main__":
    main()