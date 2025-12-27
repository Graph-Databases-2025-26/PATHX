from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from src.config.constants import DATA_DIR, GROUND_PATH

SCRIPTS_DIR = DATA_DIR.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from galois_eval import (  # noqa: E402
    _parse_table_from_obj,
    cardinality,
    f1_cell_similarity,
    tuple_constraint,
)


# --- PATHS (match exp_4_runner.py structure) ---
PRESIDENTS_DIR = DATA_DIR / "presidents"
QUERIES_PATH = PRESIDENTS_DIR / "queries_presidents.sql"
GALOIS_RESULTS_DIR = DATA_DIR / ".galois_output"
BASELINE_RESULTS_DIR = DATA_DIR / ".output"


def categorize_query_id(q_id: int) -> Tuple[str, str]:
   
    if 1 <= q_id <= 13:
        country = "Venezuela"
        if q_id in [8, 10, 13]:
            return country, "Recent"
        if q_id in [11, 12]:
            return country, "Past"
        return country, "All-Time"

    if 14 <= q_id <= 26:
        country = "USA"
        if q_id in [18, 21, 24, 26]:
            return country, "Recent"
        if q_id in [23, 25]:
            return country, "Past"
        return country, "All-Time"

    raise ValueError(f"Uncategorized query id: {q_id}")


def load_queries_map(sql_path: Path) -> Dict[int, str]:
   
    if not sql_path.exists():
        return {}

    content = sql_path.read_text(encoding="utf-8").strip()
    q_map: Dict[int, str] = {}
    idx = 1
    for q in content.split(";"):
        if not q.strip():
            continue
        lines = q.strip().splitlines()
        clean_sql = "\n".join([l for l in lines if not l.strip().startswith("--")]).strip()
        if clean_sql:
            q_map[idx] = clean_sql
            idx += 1
    return q_map


def discover_result_folders() -> List[Tuple[str, Path]]:
  
    found: List[Tuple[str, Path]] = []

    # 1) Galois
    if GALOIS_RESULTS_DIR.exists():
        for version_dir in GALOIS_RESULTS_DIR.iterdir():
            if not version_dir.is_dir():
                continue
            if "EXP5" in version_dir.name.upper():
                continue
            target = version_dir / "PRESIDENTS"
            if target.is_dir() and list(target.glob("query*.json")):
                found.append((version_dir.name.upper(), target))

    # 2) Baselines
    if BASELINE_RESULTS_DIR.exists():
        for version_dir in BASELINE_RESULTS_DIR.iterdir():
            if not version_dir.is_dir():
                continue
            if version_dir.name in ["PRESIDENTS", "FLIGHT", "GEO"]:
                continue
            target = version_dir / "watsonx" / "PRESIDENTS"
            if target.is_dir() and list(target.glob("query*.json")):
                method_name = version_dir.name.replace(".", "").replace("_output", "").upper()
                found.append((method_name, target))

    return found


def normalize_data_keys(data_list: Any) -> Any:
    """
    If we have a list of dict rows, normalize keys: "a.b" -> "b", lowercase.
    """
    if not data_list:
        return []

    if not isinstance(data_list, list):
        return data_list

    normalized: List[Any] = []
    for row in data_list:
        if not isinstance(row, dict):
            normalized.append(row)
            continue
        new_row: Dict[str, Any] = {}
        for k, v in row.items():
            clean_key = k.split(".")[-1].lower()
            new_row[clean_key] = v
        normalized.append(new_row)

    return normalized


def load_ground_truth(q_id: int) -> Any:
  
    gt_file = GROUND_PATH / "PRESIDENTS" / f"query{q_id}.json"
    if not gt_file.exists():
        return []
    try:
        data = json.loads(gt_file.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            return data.get("result_set", data.get("tuples", []))
        return []
    except Exception:
        return []


def load_results_from_folder(folder_path: Path) -> Dict[int, Any]:
    
    results_map: Dict[int, Any] = {}
    if not folder_path.exists():
        return results_map

    for p in folder_path.iterdir():
        if p.is_dir():
            continue
        name = p.name.lower()
        if not (name.endswith(".json") and name.startswith("query")):
            continue
        m = re.search(r"query(\d+)\.json", p.name, flags=re.IGNORECASE)
        if not m:
            continue
        q_id = int(m.group(1))
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                res = data.get("result_set", data.get("tuples", []))
            elif isinstance(data, list):
                res = data
            else:
                res = []
            results_map[q_id] = res
        except Exception:
            pass

    return results_map


def calculate_metrics_row(gt_data: Any, pred_data: Any) -> Tuple[float, float]:
    
    gt_clean = normalize_data_keys(gt_data)
    pr_clean = normalize_data_keys(pred_data)

    gt_cols, gt_rows = _parse_table_from_obj(gt_clean)
    pr_cols, pr_rows = _parse_table_from_obj(pr_clean)

    f1 = f1_cell_similarity(gt_cols, gt_rows, pr_cols, pr_rows)
    card = cardinality(gt_rows, pr_rows)
    tcon = tuple_constraint(gt_rows, pr_rows)
    avg = (f1 + card + tcon) / 3.0

    return avg, card


def main() -> None:
    sql_map = load_queries_map(QUERIES_PATH)
    experiments = discover_result_folders()

    if not experiments:
        print("No results found.")
        return

    all_metrics: List[Dict[str, Any]] = []

    for method_name, folder_path in experiments:
        results_map = load_results_from_folder(folder_path)

        for q_id, _sql in sql_map.items():
            # IMPORTANT: match exp_4_runner.py behavior (skip missing prediction files)
            if q_id not in results_map:
                continue

            gt_data = load_ground_truth(q_id)
            pred_data = results_map[q_id]

            avg, card = calculate_metrics_row(gt_data, pred_data)
            country, time_cat = categorize_query_id(q_id)

            all_metrics.append(
                {
                    "Method": method_name,
                    "Country": country,
                    "Time": time_cat,
                    "AVG_SCORE": avg,
                    "Cardinality": card,
                }
            )

    if not all_metrics:
        print("No metrics calculated.")
        return

    df = pd.DataFrame(all_metrics)

    print("\n" + "=" * 60)
    print("TABLE 6 REPLICATION")
    print("=" * 60)
    try:
        print(df.groupby(["Method", "Country"])["AVG_SCORE"].mean().unstack(level=0))
    except Exception as e:
        print(f"Error producing Table 6: {e}")

    print("\n" + "=" * 60)
    print("TABLE 7 REPLICATION")
    print("=" * 60)
    try:
        t7 = df.groupby(["Method", "Time"])["AVG_SCORE"].mean().unstack(level=0)
        sorter = ["Recent", "Past", "All-Time"]
        available = [s for s in sorter if s in t7.index]
        print(t7.reindex(available))
    except Exception as e:
        print(f"Error producing Table 7: {e}")


if __name__ == "__main__":
    main()
