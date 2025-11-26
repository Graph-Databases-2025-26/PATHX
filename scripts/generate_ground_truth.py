"""
Steps:
  1) Build ground truth
  2) Ingest datasets -> DuckDB files
  3) Run queries -> submissions
  4) Export EXPLAIN (txt + analyze/plan to results/)
  5) Evaluate (baseline == ground) or ground vs submissions

Examples:
  python scripts/run_all.py
  python scripts/run_all.py --datasets flight-2 flight-4 movies
  python scripts/run_all.py --skip-explain
  python scripts/run_all.py --baseline   # check ground vs submissions
"""

from __future__ import annotations
import argparse, subprocess, sys, shlex, time
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parents[1]

DEF_DATASETS = ["flight-2", "flight-4", "fortune", "geo", "movies", "premier", "presidents", "world"]
DATA_ROOT = "data"
GROUND_ROOT = "ground"
SUBMISSIONS = "submissions"            # run_queries output
EVAL_FORMAT = "table"                   # output format galois_eval.py

def run(cmd: list[str], cwd: Path = ROOT, title: str | None = None):
    t0 = time.time()
    pretty = " ".join(shlex.quote(x) for x in cmd)
    print(f"\n==== RUN: {title or pretty}\n$ {pretty}\n")
    proc = subprocess.run(cmd, cwd=str(cwd))
    dt = time.time() - t0
    if proc.returncode != 0:
        print(f"\n[ERROR] step failed ({dt:.1f}s): {title or pretty}\n", file=sys.stderr)
        sys.exit(proc.returncode)
    print(f"[OK] step finished in {dt:.1f}s: {title or pretty}")

def step_build_ground(datasets: list[str], data_root: str, ground_root: str):
    cmd = [
        PY, "scripts/build_ground_truth.py",
        "--data-root", data_root,
        "--ground-root", ground_root,
        "--datasets", *datasets,
    ]
    run(cmd, title="Build ground truth")

def step_ingest_all(datasets: list[str]):
    cmd = [PY, "src/db_interaction/ingest_all.py"]
    if datasets:
        cmd += ["--datasets", *datasets]
    run(cmd, title="Ingest datasets to DuckDB")

def step_run_queries():
    cmd = [PY, "-m", "src.pipelines.run_queries"]
    run(cmd, title="Run queries -> submissions")

def step_export_explain():
    cmd = [PY, "-m", "src.pipelines.run_explain_all"]
    run(cmd, title="Export EXPLAIN (txt/json)")

def step_evaluate(datasets_caps: list[str], ground_dir: str, submissions_dir: str, fmt: str):
    # scripts/galois_eval.py --ground <dir> --submissions <dir> --datasets
    cmd = [
        PY, "scripts/galois_eval.py",
        "--ground", ground_dir,
        "--submissions", submissions_dir,
        "--datasets", *datasets_caps,
        "--cell-metric", "similarity",
        "--tuple-metric", "constraint",
        "--overall",
        "--format", fmt,
    ]
    run(cmd, title="Evaluate submissions")

def parse_args():
    ap = argparse.ArgumentParser(description="Run full pathX pipeline end-to-end.")
    ap.add_argument("--datasets", nargs="*", default=DEF_DATASETS,
                    help=f"Datasets to process (default: {', '.join(DEF_DATASETS)})")
    ap.add_argument("--data-root", default=DATA_ROOT)
    ap.add_argument("--ground-root", default=GROUND_ROOT)
    ap.add_argument("--submissions", default=SUBMISSIONS,
                    help="Folder with submission JSONs for evaluation.")
    ap.add_argument("--baseline", action="store_true",
                    help="Evaluate ground vs ground (baseline==ground). Ignores --submissions.")
    # skip flags
    ap.add_argument("--skip-build", action="store_true")
    ap.add_argument("--skip-ingest", action="store_true")
    ap.add_argument("--skip-queries", action="store_true")
    ap.add_argument("--skip-explain", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    return ap.parse_args()

def main():
    args = parse_args()
    ds_lc = args.datasets
    ds_caps = [d.upper() for d in ds_lc]

    print(f"[INFO] Datasets: {', '.join(ds_lc)}")
    print(f"[INFO] Roots: data={args.data_root}, ground={args.ground_root}, submissions={args.submissions}")


    if not args.skip_build:
        step_build_ground(ds_lc, args.data_root, args.ground_root)

    if not args.skip_ingest:
        step_ingest_all(ds_lc)

    if not args.skip_queries:
        step_run_queries()

    if not args.skip_explain:
        step_export_explain()

    if not args.skip_eval:
        submissions_dir = args.ground_root if args.baseline else args.submissions
        step_evaluate(ds_caps, args.ground_root, submissions_dir, EVAL_FORMAT)

    print("\n All steps completed successfully.")

if __name__ == "__main__":
    main()
