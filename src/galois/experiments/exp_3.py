from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

from src.utils.tutor.galois_eval import (
    cells_set,
    _cells_similar_default,
    read_table_file,
)
from src.utils.constants import SUBMISSIONS_PATH_GALOIS, GROUND_PATH, IK_DATASETS
from src.utils.logging_config import LOG, log_init


def calculate_precision_recall(
    gt_rows: List[List[object]],
    gt_cols: List[str],
    pr_rows: List[List[object]],
    pr_cols: List[str],
) -> Tuple[float, float]:
    """
    Compute cell-level precision and recall based on the same similarity
    function used elsewhere in the project.
    """
    gt_cells = list(cells_set(gt_cols, gt_rows))
    pr_cells = list(cells_set(pr_cols, pr_rows))

    if not gt_cells and not pr_cells:
        return 1.0, 1.0
    if not gt_cells:
        return 0.0, 0.0
    if not pr_cells:
        return 1.0, 0.0

    # Precision: predicted cells that match some GT cell
    p_hits = 0
    for rc in pr_cells:
        if any(_cells_similar_default(ec, rc) for ec in gt_cells):
            p_hits += 1
    precision = p_hits / len(pr_cells)

    # Recall: GT cells that are matched by some predicted cell
    r_hits = 0
    for ec in gt_cells:
        if any(_cells_similar_default(ec, rc) for rc in pr_cells):
            r_hits += 1
    recall = r_hits / len(gt_cells)

    return precision, recall


def _load_prediction_file(path: Path):
    """
    Load rows and optional per-row scores from a prediction file.

    Supported shapes:
      - {"rows": [...], "logprobs": [...]}
      - {"result_set": [...], "logprobs": [...]}
      - {"tuples": [...], "logprobs": [...]}
      - plain list of rows: [...]
    """
    obj = json.loads(path.read_text(encoding="utf-8"))

    rows = None
    logprobs = None

    if isinstance(obj, list):
        rows = obj
    elif isinstance(obj, dict):
        if "rows" in obj:
            rows = obj.get("rows")
        elif "result_set" in obj:
            rows = obj.get("result_set")
        elif "tuples" in obj:
            rows = obj.get("tuples")
        else:
            maybe = obj.get("data")
            if isinstance(maybe, list):
                rows = maybe
        logprobs = obj.get("logprobs", None)

    if rows is None:
        rows = []

    # If scores are missing or length is inconsistent, fall back to zeros.
    if logprobs is None or len(logprobs) != len(rows):
        logprobs = [0.0] * len(rows)  # np.exp(0) = 1.0

    return rows, logprobs


def run_experiment_3():
    log_init()

    thresholds = [0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.995]
    results_map = {t: {"precisions": [], "recalls": []} for t in thresholds}

    base_pred_path = Path(SUBMISSIONS_PATH_GALOIS) / "GALOIS_F"

    LOG.info("Running threshold sweep for cell-level precision and recall.")
    LOG.info(f"Predictions root: {base_pred_path}")
    LOG.info(f"Ground-truth root: {GROUND_PATH}")
    LOG.info(f"Datasets: {IK_DATASETS}")

    total_queries = 0
    missing_logprob_warned = False

    for dataset_name in IK_DATASETS:
        ds_upper = dataset_name.upper()
        pred_dir = base_pred_path / ds_upper
        gt_dir = Path(GROUND_PATH) / ds_upper

        if not pred_dir.exists():
            LOG.warning(f"Missing prediction directory for {dataset_name}: {pred_dir}")
            continue
        if not gt_dir.exists():
            LOG.warning(f"Missing ground-truth directory for {dataset_name}: {gt_dir}")
            continue

        LOG.info(f"Processing dataset: {dataset_name}")

        for json_file in sorted(pred_dir.glob("query*.json")):
            rows, logprobs = _load_prediction_file(json_file)
            if not rows:
                continue

            stem = json_file.stem  # e.g. "query12"
            digits = "".join(ch for ch in stem if ch.isdigit())
            if not digits:
                continue
            q_id = int(digits)

            gt_json = gt_dir / f"query{q_id}.json"
            gt_csv = gt_dir / f"query{q_id}.csv"

            if gt_json.exists():
                gt_path = gt_json
            elif gt_csv.exists():
                gt_path = gt_csv
            else:
                LOG.warning(f"No ground truth for {dataset_name}, query {q_id}")
                continue

            gt_cols, gt_rows_list = read_table_file(gt_path)

            if not missing_logprob_warned and all(lp == 0.0 for lp in logprobs):
                LOG.warning(
                    "Per-row scores not found in predictions; "
                    "using a default score of 1.0 for all rows."
                )
                missing_logprob_warned = True

            for t in thresholds:
                filtered_rows = []
                for row, lp in zip(rows, logprobs):
                    # If lp <= 0: treat as log-prob; else treat as probability.
                    prob = float(np.exp(lp)) if lp <= 0 else float(lp)
                    if prob >= t:
                        filtered_rows.append(row)

                if filtered_rows:
                    pr_cols = list(filtered_rows[0].keys())
                    pr_rows_list = [[r.get(c, "") for c in pr_cols] for r in filtered_rows]
                else:
                    pr_cols = []
                    pr_rows_list = []

                p, r = calculate_precision_recall(
                    gt_rows_list, gt_cols, pr_rows_list, pr_cols
                )
                results_map[t]["precisions"].append(p)
                results_map[t]["recalls"].append(r)

            total_queries += 1

    LOG.info(f"Total queries processed: {total_queries}")

    if total_queries == 0:
        LOG.error("No queries processed. Check directories and dataset list.")
        return

    thresholds_sorted = sorted(thresholds)
    avg_precisions = []
    avg_recalls = []

    for t in thresholds_sorted:
        p_list = results_map[t]["precisions"]
        r_list = results_map[t]["recalls"]

        avg_p = float(np.mean(p_list)) if p_list else 0.0
        avg_r = float(np.mean(r_list)) if r_list else 0.0

        avg_precisions.append(avg_p)
        avg_recalls.append(avg_r)

        LOG.info(
            "Threshold %.3f -> precision=%.3f, recall=%.3f (n=%d)",
            t,
            avg_p,
            avg_r,
            len(p_list),
        )

    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.size"] = 11

    fig, ax = plt.subplots(figsize=(8, 3.5))
    x_idx = np.arange(len(thresholds_sorted))

    ax.plot(
        x_idx,
        avg_precisions,
        marker="o",
        linestyle="-",
        linewidth=1.8,
        markersize=5,
        label="Precision",
    )
    ax.plot(
        x_idx,
        avg_recalls,
        marker="s",
        linestyle="-",
        linewidth=1.8,
        markersize=5,
        label="Recall",
    )

    ax.set_xticks(x_idx)
    ax.set_xticklabels([f"{t:.3f}".rstrip("0").rstrip(".") for t in thresholds_sorted])
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.6)
    legend = ax.legend(loc="lower left", ncol=2, frameon=True, fontsize=9)
    legend.get_frame().set_linewidth(0.8)

    plt.tight_layout()
    out_path = Path("threshold_precision_recall.png")
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    LOG.info("Figure saved as %s", out_path.resolve())


if __name__ == "__main__":
    run_experiment_3()
