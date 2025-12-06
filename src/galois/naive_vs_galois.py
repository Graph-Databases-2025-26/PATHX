from __future__ import annotations

import time
from typing import Any, Dict, List

from src.galois.run_galois import GaloisRunner


NAIVE_PROMPT = """
You are an expert knowledgeable about real-world data.

Answer the following question using your world knowledge.

Question:
"What are size and population of European cities with more than 1M people and more than fifteen private hospitals?"

Output format:
- Provide a list of cities.
- Use one line per city in the following format:
  City | Country | Population | Number of private hospitals
- Do NOT return JSON.
- Do NOT provide explanations, comments, or reasoning.
- Just list the results, one per line.
"""


def _call_llm(llm, prompt: str) -> tuple[str, float]:
    """Call the underlying LLM and return (content, elapsed_seconds)."""
    t0 = time.time()
    raw = llm.invoke(prompt)
    elapsed = time.time() - t0

    # LangChain / OpenAI-style message objects usually have `.content`
    content = getattr(raw, "content", str(raw))
    return content, elapsed


def _parse_naive_output(text: str) -> List[Dict[str, Any]]:
    """
    Parse the naive bullet/line output into rows.

    Expected format per line (as enforced by the prompt):

        City | Country | Population | Number of private hospitals

    We keep population / hospitals as strings; this is just for display.
    """
    rows: List[Dict[str, Any]] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        # allow optional leading bullet like "- " or "• "
        if line.startswith(("-", "•")):
            line = line[1:].strip()

        # We only care about lines that look like a table row
        if "|" not in line:
            continue

        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            # Not enough columns, skip
            continue

        city, country, population, num_hosp = parts[:4]

        rows.append(
            {
                "name": city,
                "country": country,
                "population": population,
                "num_private_hospitals": num_hosp,
            }
        )

    # If parsing completely failed, fall back to a single text column
    if not rows:
        rows = [{"text": line} for line in text.splitlines() if line.strip()]

    return rows


class NaiveLLMRunner:
    """Very simple NL baseline – one direct question to the LLM."""

    def __init__(self, llm):
        self.llm = llm

    def run_eu_cities(self) -> Dict[str, Any]:
        """
        Ask the naive NL question and return a dict in the same shape
        as Galois_* results, so it can be passed into compare_results().
        """
        content, elapsed = _call_llm(self.llm, NAIVE_PROMPT)
        rows = _parse_naive_output(content)

        # We don't have token accounting here; set to 0 for compatibility.
        return {
            "result_set": rows,
            "time": round(elapsed, 2),
            "tokens": 0,
            "meta": {
                "strategy": "naive_nl",
                "raw_output": content[:5000],  # truncated raw text, if needed
            },
        }


def run_eu_cities() -> None:
    # Use the same config + LLM as the regular Galois runner
    runner = GaloisRunner()

    print("\n==================== Naive vs GALOIS (EU_Cities) ====================\n")

    # 1) Run naive NL baseline
    naive = NaiveLLMRunner(runner.llm)
    naive_result = naive.run_eu_cities()

    # 2) Run all Galois variants on the same logical query
    galois_results = runner.run_query(
        dataset="WORLD",
        table="EU_Cities",
        attributes=["name", "country", "population", "num_private_hospitals"],
        where_clause="population > 1M AND num_private_hospitals > 15",
        variants_to_run=["Galois_WO", "Galois_A", "Galois_S", "Galois_F"],
    )

    # 3) Combine results: add naive baseline as another "variant"
    combined_results: Dict[str, Any] = {"Naive_NL": naive_result}
    combined_results.update(galois_results)

    # 4) Print comparison summary & tables (reusing existing logic)
    runner.compare_results(combined_results)

    # 5) Optionally save combined results for later analysis
    from pathlib import Path
    import json

    out_path = Path("src/test_results/naive_vs_galois_eu_cities.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=2, ensure_ascii=False)
    print(f"\nCombined results saved to: {out_path}")



if __name__ == "__main__":
    run_eu_cities()
