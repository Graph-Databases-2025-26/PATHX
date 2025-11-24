import subprocess
from tabulate import tabulate
import pandas as pd
import io
import os
from pathlib import Path

GALOIS_SCRIPT_PATH = Path("scripts") / "galois_eval.py"
GROUND_TRUTH_ROOT = "ground"
SQL_SUBMISSIONS_ROOT = "sql_baseline_results"
NL_SUBMISSIONS_ROOT = "nl_baseline_results"

PAPER_RESULTS = {
    "NL": {
        "F1-Cell": 0.237,
        "Cardinality": 0.462,
        "Tuple Constr.": 0.065,
        "AVG-Score": 0.254,
        "#Tokens (M)": 0.83,
        "Avg Time": 120.0,
    },
    "SQL": {
        "F1-Cell": 0.431,
        "Cardinality": 0.659,
        "Tuple Constr.": 0.351,
        "AVG-Score": 0.481,
        "#Tokens (M)": 0.33,
        "Avg Time": 61.4,
    },
}

METRICS_ORDER = [
    "F1-Cell",
    "Cardinality",
    "Tuple Constr.",
    "AVG-Score",
    "#Tokens (M)",
    "Avg Time",
]


def run_galois_evaluation(submissions_path: str, baseline_name: str) -> dict:
    """
    Runs the galois_eval.py script on the specified submissions path
    and returns a dictionary of evaluation metrics.
    """
    if not os.path.exists(GALOIS_SCRIPT_PATH):
        print(f"ERROR: GALOIS script not found at {GALOIS_SCRIPT_PATH}")
        return {}
    if not os.path.isdir(submissions_path):
        print(f"WARNING: Submissions folder '{baseline_name}' not found at {submissions_path}. Skipping.")
        return {}

    command = [
        "python",
        str(GALOIS_SCRIPT_PATH),
        "--ground",
        str(GROUND_TRUTH_ROOT),
        "--submissions",
        str(submissions_path),
        "--cell-metric",
        "similarity",
        "--tuple-metric",
        "similarity",
        "--overall",
        "--format",
        "csv",
    ]

    print(f"-> Evaluating baseline {baseline_name}...")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding="utf-8",
        )
        df = pd.read_csv(io.StringIO(result.stdout))

        if df.empty or df.iloc[0]["Dataset"] != "ALL":
            print(f"ERROR: galois_eval.py did not return aggregated results for {baseline_name}.")
            return {}

        all_results = df.iloc[0].to_dict()
        our_results = {}

        for metric in ["F1-Cell", "Cardinality", "Tuple Constr.", "AVG-Score"]:
            our_results[metric] = float(all_results.get(metric, 0.0))

        # Tokens in Millions
        tokens_val = all_results.get("#Tokens")
        our_results["#Tokens (M)"] = float(tokens_val) / 1_000_000 if tokens_val is not None else 0.0

        # Avg Time
        our_results["Avg Time"] = float(all_results.get("Avg Time (s)", 0.0))

        print(f"-> Successfully retrieved {baseline_name} results.")
        return our_results

    except subprocess.CalledProcessError as e:
        print(f"\nERROR: galois_eval.py failed for {baseline_name}. Error: {e.stderr}")
        return {}
    except Exception as e:
        print(f"\nERROR: Unexpected error while processing {baseline_name}: {e}")
        return {}


def generate_report_table(our_nl_results: dict, our_sql_results: dict):
    """
    Generates and prints a comparison report table between paper results and our results.
    """
    report_data = []

    for metric in METRICS_ORDER:
        paper_nl = PAPER_RESULTS["NL"].get(metric)
        paper_sql = PAPER_RESULTS["SQL"].get(metric)
        our_nl = our_nl_results.get(metric, 0.0)
        our_sql = our_sql_results.get(metric, 0.0)

        # Absolute deltas
        delta_nl = (our_nl - paper_nl) if paper_nl is not None else None
        delta_sql = (our_sql - paper_sql) if paper_sql is not None else None

        # Relative deltas
        rel_nl = ((our_nl - paper_nl) / paper_nl * 100.0) if (paper_nl not in (None, 0)) else None
        rel_sql = ((our_sql - paper_sql) / paper_sql * 100.0) if (paper_sql not in (None, 0)) else None

        def fmt_val(val, metric_name: str, num_decimals=3):
            """
            Formats the metric value for display.
            """
            if val is None:
                return ""
            if metric_name in ("#Tokens (M)", "Avg Time"):
                num_decimals = 2
            return f"{val:,.{num_decimals}f}"

        def fmt_pct(val):
            """
            Formats the percentage value for display.
            """
            if val is None:
                return ""
            return f"{val:+.1f}%"

        report_data.append(
            {
                "METRIC": metric,
                "NL (Paper)": fmt_val(paper_nl, metric),
                "SQL (Paper)": fmt_val(paper_sql, metric),
                "NL": fmt_val(our_nl, metric),
                "SQL": fmt_val(our_sql, metric),
                "Δ NL": fmt_val(delta_nl, metric),
                "Δ SQL": fmt_val(delta_sql, metric),
                "Δ NL (%)": fmt_pct(rel_nl),
                "Δ SQL (%)": fmt_pct(rel_sql),
            }
        )

    df_report = pd.DataFrame(report_data)

    print(
        tabulate(
            df_report,
            headers="keys",
            tablefmt="psql",
            showindex=False,
            floatfmt=".3f",
        )
    )


if __name__ == "__main__":

    # Run evaluations for NL baseline
    our_nl_results = run_galois_evaluation(NL_SUBMISSIONS_ROOT, "NL")

    if not our_nl_results:
        print("\nUsing default NL results (0.0) for report generation.")
        our_nl_results = {m: 0.0 for m in METRICS_ORDER}

    # Run evaluations for SQL baseline
    our_sql_results = run_galois_evaluation(SQL_SUBMISSIONS_ROOT, "SQL")

    if not our_sql_results:
        print("Using default SQL results (0.0) for report generation.")
        our_sql_results = {m: 0.0 for m in METRICS_ORDER}

    # Generate and print the report table
    generate_report_table(our_nl_results, our_sql_results)
