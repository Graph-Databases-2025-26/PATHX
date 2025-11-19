import subprocess
import sys
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]

NL_MODULE = "src.pipelines.nl_baseline"
SQL_MODULE = "src.pipelines.sql_baseline"
COMPARE_MODULE = "src.pipelines.compare_results"

PROVIDER_CHOICES = ("azure", "gemini", "watsonx")


def run_step(name: str, command: list[str]):
    print("\n" + "=" * 80)
    print(f"▶ Running step: {name}")
    print("=" * 80)
    print("$", " ".join(command))

    try:
        subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            check=True,
        )
        print(f"\n✓ {name} completed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"\n✗ ERROR during '{name}' (exit code {e.returncode})")
        sys.exit(e.returncode)


def build_nl_command(args: argparse.Namespace) -> list[str]:
    cmd = ["python", "-m", NL_MODULE]

    if args.datasets:
        cmd += ["--datasets", *args.datasets]

    if args.nl_provider:
        cmd += ["--provider", args.nl_provider]

    if args.nl_strategy:
        cmd += ["--strategy", args.nl_strategy]

    return cmd


def build_sql_command(args: argparse.Namespace) -> list[str]:
    cmd = ["python", "-m", SQL_MODULE]

    if args.datasets:
        cmd += ["--datasets", *args.datasets]

    if args.sql_provider:
        cmd += ["--provider", args.sql_provider]

    if args.sql_strategy:
        cmd += ["--strategy", args.sql_strategy]

    return cmd


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Run NL baseline, SQL baseline, and comparison."
    )

    # dataset selection
    ap.add_argument(
        "--datasets",
        nargs="*",
        help="Datasets to run (folder names under ./data).",
    )

    # model selection
    ap.add_argument(
        "--nl-provider",
        choices=PROVIDER_CHOICES,
        default="watsonx",
        help="Provider for NL baseline.",
    )
    ap.add_argument(
        "--sql-provider",
        choices=PROVIDER_CHOICES,
        default="watsonx",
        help="Provider for SQL baseline.",
    )

    # strategy selection
    ap.add_argument(
        "--nl-strategy",
        help="Strategy for NL baseline (if supported).",
    )
    ap.add_argument(
        "--sql-strategy",
        help="Strategy for SQL baseline (direct / two_step / oracle / ...).",
    )

    # skip flags
    ap.add_argument("--skip-nl", action="store_true", help="Skip NL baseline.")
    ap.add_argument("--skip-sql", action="store_true", help="Skip SQL baseline.")
    ap.add_argument("--skip-compare", action="store_true", help="Skip comparison.")

    # comparison-only mode
    ap.add_argument(
        "--compare-only",
        action="store_true",
        help="Run ONLY compare_results.py (skip both NL and SQL).",
    )

    return ap


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    # Только сравнение
    if args.compare_only:
        print("\n[INFO] Running in compare-only mode.")
        run_step("Compare Results", ["python", "-m", COMPARE_MODULE])
        return

    # NL baseline
    if not args.skip_nl:
        nl_cmd = build_nl_command(args)
        run_step("NL Baseline", nl_cmd)
    else:
        print("\n[INFO] Skipping NL baseline.")

    # SQL baseline
    if not args.skip_sql:
        sql_cmd = build_sql_command(args)
        run_step("SQL Baseline", sql_cmd)
    else:
        print("\n[INFO] Skipping SQL baseline.")

    # Compare
    if not args.skip_compare:
        run_step("Compare Results", ["python", "-m", COMPARE_MODULE])
    else:
        print("\n[INFO] Skipping compare_results step.")


if __name__ == "__main__":
    main()
