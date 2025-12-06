"""
Quick experiment:
compare a naive single-call LLM query vs. the Galois variants (WO, A, S, F)
on the EU_Cities query from the paper.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd
from tabulate import tabulate

from src.galois.run_galois import GaloisRunner
from src.galois.operators.base_scan import BaseScanOperator
from src.galois.prompts.templates import PromptTemplates
from src.galois.utils.response_parser import ResponseParser


class NaiveLLMRunner(BaseScanOperator):
    """
    Thin wrapper that reuses BaseScanOperator utilities (_call_llm, _get_content)
    to run a *single* naive LLM query and still measure time & tokens.
    """

    def __init__(self, llm):
        super().__init__(llm, max_iters=1)

    def scan(
        self,
        dataset: str,
        table: str,
        attributes: List[str],
        where_clause: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self.run_naive(dataset, table, attributes, where_clause)


    def run_naive(
        self,
        dataset: str,
        table: str,
        attributes: List[str],
        where_clause: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Ask the LLM directly for rows that satisfy the WHERE clause in one call.

        Returns:
            Dict in the same shape as Galois_* operators:
            {
              "result_set": [...],
              "time": float,
              "tokens": int,
              "meta": {...}
            }
        """
        # Reuse our table-scan prompt (single shot, no follow-ups)
        prompt = PromptTemplates.table_scan_first(
            dataset=dataset,
            table=table,
            attributes=attributes,
            where_clause=where_clause,
        )

        raw, elapsed, tokens = self._call_llm(prompt)
        content = self._get_content(raw)
        parsed = ResponseParser.parse_llm_response(content, elapsed, tokens)

        result_set = parsed.get("result_set", [])

        return {
            "result_set": result_set,
            "time": round(elapsed, 3),
            "tokens": tokens,
            "meta": {
                "strategy": "naive_single_call",
                "variant": "Naive_LLM",
            },
        }


def run_eu_cities() -> Dict[str, Any]:
    """
    Run the EU_Cities query both:
      1) in a naive single LLM call
      2) through all Galois variants

    and print a small comparison table.
    """
    # GaloisRunner will load config + create LLM instance
    runner = GaloisRunner()
    naive = NaiveLLMRunner(runner.llm)

    dataset = "WORLD"
    table = "EU_Cities"
    attributes = ["name", "country", "population", "num_private_hospitals"]
    where_clause = "population > 1M AND num_private_hospitals > 15"

    print("Naive vs Galois: EU_CITIES query")

    # 1) Naive single-call baseline
    print("\nRunning Naive_LLM...")
    naive_result = naive.run_naive(dataset, table, attributes, where_clause)
    print(
        f"   {len(naive_result['result_set'])} rows, "
        f"{naive_result['tokens']:,} tokens, "
        f"{naive_result['time']:.2f}s"
    )

    # 2) Galois variants
    galois_results = runner.run_query(
        dataset=dataset,
        table=table,
        attributes=attributes,
        where_clause=where_clause,
        variants_to_run=["Galois_WO", "Galois_A", "Galois_S", "Galois_F"],
    )

    # Combine all results into one dict for a joint summary
    combined: Dict[str, Any] = {"Naive_LLM": naive_result}
    combined.update(galois_results)

    # -------- COMBINED SUMMARY (naive + all Galois) --------
    print("\n\n=== COMBINED SUMMARY (Naive vs Galois) ===")

    rows = []
    for variant_name, result in combined.items():
        if "error" in result:
            rows.append(
                {
                    "Variant": variant_name,
                    "Rows": "ERROR",
                    "Tokens": "",
                    "Time (s)": "",
                    "Strategy": result["error"][:40]
                    + ("..." if len(result["error"]) > 40 else ""),
                }
            )
            continue

        rows.append(
            {
                "Variant": variant_name,
                "Rows": len(result.get("result_set", [])),
                "Tokens": result.get("tokens", 0),
                "Time (s)": result.get("time", 0.0),
                "Strategy": result.get("meta", {}).get("strategy", "N/A"),
            }
        )

    df = pd.DataFrame(rows)
    print(
        tabulate(
            df,
            headers="keys",
            tablefmt="psql",
            showindex=False,
            floatfmt=".2f",
        )
    )

    # Optional: reuse the existing pretty printer for Galois-only breakdown
    print("\n\n=== Galois-only breakdown ===")
    runner.compare_results(
        galois_results,
        extra_results={"Naive_LLM": naive_result},
    )

    return {"naive": naive_result, "galois": galois_results}


if __name__ == "__main__":
    run_eu_cities()
