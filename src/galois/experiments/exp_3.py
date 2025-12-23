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


