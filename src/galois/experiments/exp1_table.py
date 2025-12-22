#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


METRICS: List[Tuple[str, str]] = [
    ("F1-Cell", "F1-Cell"),
    ("Cardinality", "Cardinality"),
    ("Tuple Constr.", "Tuple-Constraint"),
    ("AVG-Score", "AVG-Score"),
    ("#Tokens (M)", "#Tokens"),
    ("Avg Time (s)", "Avg-Time"),
]


def _fmt(x: Any, *, metric_key: str) -> str:
    if x is None:
        return "—"
    try:
        if metric_key == "#Tokens":
            # evaluator returns raw tokens; show in millions
            return f"{float(x) / 1_000_000:.3f}"
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def _ascii_table(headers: List[str], rows: List[List[str]]) -> None:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def sep():
        print("+" + "+".join("─" * (w + 2) for w in widths) + "+")

    def row(cells: List[str]):
        print("| " + " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells))) + " |")

    sep()
    row(headers)
    sep()
    for r in rows:
        row(r)
    sep()


def _run_eval_json(
    *,
    eval_script: Path,
    ground: Path,
    submissions: Path,
    datasets: List[str],
    cell_metric: str,
    tuple_metric: str,
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(eval_script),
        "--ground",
        str(ground),
        "--submissions",
        str(submissions),
        "--cell-metric",
        cell_metric,
        "--tuple-metric",
        tuple_metric,
        "--overall",
        "--format",
        "json",
    ]
    if datasets:
        cmd += ["--datasets", *datasets]

    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        msg = (p.stderr or p.stdout or "").strip()
        raise RuntimeError(f"galois_eval failed for submissions='{submissions}': {msg}")

    try:
        obj = json.loads(p.stdout)
    except Exception as e:
        raise RuntimeError(
            f"Failed to parse JSON from galois_eval output for submissions='{submissions}': {e}\n"
            f"STDOUT was:\n{p.stdout[:1000]}"
        )

    if "ALL" not in obj or not isinstance(obj["ALL"], dict):
        raise RuntimeError(f"galois_eval JSON missing ALL row for submissions='{submissions}'")

    return obj["ALL"]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Exp-1 pretty table: runs scripts/galois_eval.py internally and prints a nice table."
    )
    ap.add_argument("--ground", type=Path, required=True, help="Path to ground truth folder (contains dataset subfolders).")
    ap.add_argument(
        "--submissions",
        type=Path,
        nargs="+",
        required=True,
        help="One or more submissions folders (e.g. results_galois_wo results_galois_a ...).",
    )
    ap.add_argument("--datasets", nargs="*", default=None, help="Datasets to include (default: all under --ground).")
    ap.add_argument("--cell-metric", choices=["exact", "similarity"], default="exact")
    ap.add_argument("--tuple-metric", choices=["constraint", "similarity"], default="constraint")
    ap.add_argument(
        "--eval-script",
        type=Path,
        default=Path("scripts/galois_eval.py"),
        help="Path to galois_eval.py (default: scripts/galois_eval.py).",
    )

    args = ap.parse_args()

    if not args.eval_script.exists():
        raise SystemExit(f"eval script not found: {args.eval_script}")
    if not args.ground.exists():
        raise SystemExit(f"ground not found: {args.ground}")

    # Resolve dataset list if not provided
    datasets = args.datasets or []

    # Build columns (one per submissions folder)
    # Use folder name as column header
    col_names = [p.name for p in args.submissions]

    # Run evaluator for each submissions folder
    results: Dict[str, Dict[str, Any]] = {}
    for sub in args.submissions:
        if not sub.exists():
            raise SystemExit(f"submissions not found: {sub}")

        all_row = _run_eval_json(
            eval_script=args.eval_script,
            ground=args.ground,
            submissions=sub,
            datasets=datasets,
            cell_metric=args.cell_metric,
            tuple_metric=args.tuple_metric,
        )
        results[sub.name] = all_row

    # Prepare pretty table rows
    headers = ["Metric", *col_names]
    rows: List[List[str]] = []

    for label, key in METRICS:
        r = [label]
        for cn in col_names:
            r.append(_fmt(results[cn].get(key), metric_key=key))
        rows.append(r)

    _ascii_table(headers, rows)


if __name__ == "__main__":
    main()
