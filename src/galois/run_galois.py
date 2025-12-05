"""
Main script to run and compare all Galois variants
"""
from __future__ import annotations

import json
from typing import Any, Dict, Optional
from tabulate import tabulate
import pandas as pd
from pathlib import Path

from src.baselines.llm_factory import get_llm_wrapper
from src.galois.variants.galois_wo import GaloisWOOperator
from src.galois.variants.galois_a import GaloisA
from src.galois.variants.galois_s import GaloisS
from src.galois.variants.galois_f import GaloisF
from src.config.config import load_config, GaloisConfig


class GaloisRunner:
    """
    Runner to execute and compare all Galois variants.
    """

    def __init__(self, config: Optional[GaloisConfig] = None):
        """
        Args:
            config: GaloisConfig instance. If None, loads from .env
        """
        if config is None:
            config = load_config()
        
        self.config = config
        
        # Get LLM wrapper
        wrapper = get_llm_wrapper(config)
        self.llm = wrapper.get_llm_instance()
        
        # Initialize all variants
        self.variants = {
            "Galois_WO": GaloisWOOperator(self.llm, max_iters=5),
            "Galois_A": GaloisA(self.llm, max_iters=5),
            "Galois_S": GaloisS(self.llm, max_iters=5),
            "Galois_F": GaloisF(
                self.llm,
                confidence_threshold = config.galois.confidence_threshold,
                max_iters = config.galois.max_iterations,
            ),
        }

    def run_query(
        self,
        dataset: str,
        table: str,
        attributes: list[str],
        where_clause: str | None = None,
        variants_to_run: list[str] | None = None,
    ) -> Dict[str, Any]:
        """
        Run query through specified Galois variants.
        
        Args:
            dataset: Dataset name (e.g., "WORLD")
            table: Table name (e.g., "city")
            attributes: List of attribute names
            where_clause: SQL WHERE clause (without "WHERE")
            variants_to_run: List of variant names to run (default: all)
        
        Returns:
            Dict mapping variant_name -> result
        """
        if variants_to_run is None:
            variants_to_run = list(self.variants.keys())
        
        results = {}
        
        print(f"\n{'='*70}")
        print(f"Running Query:")
        print(f"  Dataset: {dataset}")
        print(f"  Table: {table}")
        print(f"  Attributes: {', '.join(attributes)}")
        print(f"  WHERE: {where_clause or '(none)'}")
        print(f"{'='*70}\n")
        
        for variant_name in variants_to_run:
            if variant_name not in self.variants:
                print(f"Warning: Unknown variant: {variant_name}")
                continue
            
            print(f"Running {variant_name}...")
            
            try:
                variant = self.variants[variant_name]
                
                # Execute based on variant type
                if isinstance(variant, GaloisWOOperator):
                    result = variant.scan(dataset, table, attributes, where_clause)
                else:
                    result = variant.execute(dataset, table, attributes, where_clause)
                
                results[variant_name] = result
                
                print(f"   {len(result['result_set'])} rows, "
                      f"{result['tokens']:,} tokens, "
                      f"{result['time']:.2f}s")
            
            except Exception as e:
                print(f"   Error: {e}")
                results[variant_name] = {"error": str(e)}
        
        return results

    def compare_results(self, results: Dict[str, Any]) -> None:
            """
            Print comparison table of results (similar to compare_results.py).
            """
            # -------- SUMMARY TABLE --------
            summary_rows = []

            for variant_name, result in results.items():
                if "error" in result:
                    summary_rows.append(
                        {
                            "Variant": variant_name,
                            "Rows": "ERROR",
                            "Tokens": "",
                            "Time (s)": "",
                            "Strategy": result["error"][:40] + ("..." if len(result["error"]) > 40 else ""),
                        }
                    )
                    continue

                strategy = result.get("meta", {}).get("strategy", "N/A")
                summary_rows.append(
                    {
                        "Variant": variant_name,
                        "Rows": len(result.get("result_set", [])),
                        "Tokens": result.get("tokens", 0),
                        "Time (s)": result.get("time", 0.0),
                        "Strategy": strategy,
                    }
                )

            df_summary = pd.DataFrame(summary_rows)

            print("\nCOMPARISON SUMMARY")
            print(
                tabulate(
                    df_summary,
                    headers="keys",
                    tablefmt="psql",
                    showindex=False,
                    floatfmt=".2f",
                )
            )

            # -------- RESULT SET CONTENT --------
            print("\nRESULT SET CONTENT")

            for variant_name, result in results.items():
                print(f"\n=== {variant_name} ===")

                if "error" in result:
                    print(f"ERROR: {result['error']}")
                    continue

                result_set = result.get("result_set", [])

                if not result_set:
                    print("(empty result set)")
                    continue

                # result_set: list of dicts -> DataFrame
                if isinstance(result_set[0], dict):
                    df_rs = pd.DataFrame(result_set)
                else:
                    # Fallback if it's just a list of values/tuples
                    df_rs = pd.DataFrame({"Row": result_set})

                print(
                    tabulate(
                        df_rs,
                        headers="keys",
                        tablefmt="psql",
                        showindex=False,
                    )
                )

            # -------- TOKEN COST COMPARISON --------
            baseline_name = "Galois_WO"
            if baseline_name in results and "error" not in results[baseline_name]:
                baseline_tokens = results[baseline_name].get("tokens", 0) or 0

                token_rows = []
                for variant_name, result in results.items():
                    if variant_name == baseline_name or "error" in result:
                        continue

                    tokens = result.get("tokens", 0) or 0
                    ratio = tokens / baseline_tokens if baseline_tokens > 0 else 0.0
                    reduction = (1 - ratio) * 100 if baseline_tokens > 0 else 0.0

                    token_rows.append(
                        {
                            "Variant": variant_name,
                            "Tokens": tokens,
                            "Rel. cost": f"{ratio:.1%}",
                            "Reduction": f"{reduction:+.1f}%",
                        }
                    )

                if token_rows:
                    df_tokens = pd.DataFrame(token_rows)
                    print("\nTOKEN COST RELATIVE TO Galois_WO")
                    print(
                        tabulate(
                            df_tokens,
                            headers="keys",
                            tablefmt="psql",
                            showindex=False,
                        )
                    )


    def save_results(self, results: Dict[str, Any], filepath: str) -> None:
        """Save results to JSON file"""
        path = Path(filepath)                     
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open('w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: {path}")

def run_eu_cities(runner: GaloisRunner) -> None:
    print("\n" + "=" * 70)
    print("EU_CITIES QUERY (multi-condition WHERE)")
    print("=" * 70)

    results = runner.run_query(
        dataset="WORLD",
        table="EU_Cities",
        attributes=["name", "country", "population", "num_private_hospitals"],
        where_clause="population > 1M AND num_private_hospitals > 15",
    )

    runner.compare_results(results)
    runner.save_results(results, "src/test_results/galois_results_eu_cities.json")


def run_geo_states(runner: GaloisRunner) -> None:
    print("\n" + "=" * 70)
    print("GEO.STATES QUERY (single-condition WHERE)")
    print("=" * 70)

    results = runner.run_query(
        dataset="GEO",
        table="states",
        attributes=["name", "area", "population", "capital"],
        where_clause="area > 300000",
    )

    runner.compare_results(results)
    runner.save_results(results, "src/test_results/galois_results_geo_states.json")


def main() -> None:

    runner = GaloisRunner()

    run_eu_cities(runner)
    # run_geo_states(runner)


if __name__ == "__main__":
    main()
