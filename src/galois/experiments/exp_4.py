import re
import pandas as pd
import json
import os
import glob
import sys
from src.utils import LOG
from src.utils.constants import GROUND_PATH, DATA_DIR

# --- PATH SETUP ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../../../'))
EVAL_DIR = os.path.join(PROJECT_ROOT, 'src', 'utils', 'tutor')

if EVAL_DIR not in sys.path:
    sys.path.append(EVAL_DIR)

sys.path.append(SCRIPT_DIR)

try:
    from galois_eval import (
        _parse_table_from_obj,
        f1_cell_similarity,
        cardinality,
        tuple_constraint
    )
except ImportError:
    LOG.error(f"CRITICAL ERROR: Cannot find `galois_eval.py` in: {SCRIPT_DIR}.")
    sys.exit(1)

# Paths
PRESIDENTS_DIR = os.path.join(DATA_DIR, "PRESIDENTS")
QUERIES_PATH = os.path.join(PRESIDENTS_DIR, "queries_presidents.sql")
GALOIS_RESULTS_DIR = os.path.join(DATA_DIR, ".galois_output")
BASELINE_RESULTS_DIR = os.path.join(DATA_DIR, ".output")


# --- LOGIC ---

def categorize_query_id(q_id):
    """
    Map query id to (Country, Time) category.

    Time categories (as in the Galois paper experiment 4):
        - 'Recent'
        - 'Past'
        - 'All-Time'
    Countries:
        - 'USA'
        - 'UK'
        - 'France'
    """
    # Example mapping based on query id ranges.
    # Adjust if your query IDs differ, but this matches the original reference implementation.

    if 1 <= q_id <= 10:
        country = "USA"
        time = "All-Time"
    elif 11 <= q_id <= 20:
        country = "USA"
        time = "Recent"
    elif 21 <= q_id <= 30:
        country = "USA"
        time = "Past"
    elif 31 <= q_id <= 40:
        country = "UK"
        time = "All-Time"
    elif 41 <= q_id <= 50:
        country = "UK"
        time = "Recent"
    elif 51 <= q_id <= 60:
        country = "UK"
        time = "Past"
    elif 61 <= q_id <= 70:
        country = "France"
        time = "All-Time"
    elif 71 <= q_id <= 80:
        country = "France"
        time = "Recent"
    elif 81 <= q_id <= 90:
        country = "France"
        time = "Past"
    else:
        country = "Unknown"
        time = "Unknown"

    return country, time


def load_queries_map(file_path):
    q_map = {}
    idx = 1
    if not os.path.exists(file_path):
        LOG.warning(f"Query file not found at {file_path}")
        return {}

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
        for q in content.split(";"):
            if q.strip():
                lines = q.strip().split('\n')
                clean_sql = '\n'.join([l for l in lines if not l.strip().startswith('--')]).strip()
                if clean_sql:
                    q_map[idx] = clean_sql
                    idx += 1
    return q_map


def discover_result_folders():
    found_folders = []

    # 1. Search GALOIS
    if os.path.exists(GALOIS_RESULTS_DIR):
        for version in os.listdir(GALOIS_RESULTS_DIR):
            if "EXP5" in version.upper():
                continue
            target_path = os.path.join(GALOIS_RESULTS_DIR, version, "PRESIDENTS")
            if os.path.isdir(target_path):
                if glob.glob(os.path.join(target_path, "query*.json")):
                    found_folders.append((version.upper(), target_path))

    # 2. Search BASELINES
    if os.path.exists(BASELINE_RESULTS_DIR):
        for version_folder in os.listdir(BASELINE_RESULTS_DIR):
            if version_folder in ['PRESIDENTS', 'FLIGHT', 'GEO']:
                continue
            target_path = os.path.join(BASELINE_RESULTS_DIR, version_folder, "watsonx", "PRESIDENTS")
            if os.path.isdir(target_path):
                if glob.glob(os.path.join(target_path, "query*.json")):
                    method_name = version_folder.replace(".", "").replace("_output", "").upper()
                    found_folders.append((method_name, target_path))

    return found_folders


def normalize_data_keys(data_list):
    if not data_list:
        return []
    normalized = []
    for row in data_list:
        if not isinstance(row, dict):
            normalized.append(row)
            continue
        new_row = {}
        for k, v in row.items():
            clean_key = k.split('.')[-1].lower()
            new_row[clean_key] = v
        normalized.append(new_row)
    return normalized


def calculate_metrics_row(gt_data, pred_data):
    gt_clean = normalize_data_keys(gt_data)
    pred_clean = normalize_data_keys(pred_data)

    gt_cols, gt_rows = _parse_table_from_obj(gt_clean)
    pr_cols, pr_rows = _parse_table_from_obj(pred_clean)

    f1 = f1_cell_similarity(gt_cols, gt_rows, pr_cols, pr_rows)
    card = cardinality(gt_rows, pr_rows)
    tcon = tuple_constraint(gt_rows, pr_rows)

    avg = (f1 + card + tcon) / 3.0
    return avg, card


def load_ground_truth(q_id):
    gt_file = os.path.join(GROUND_PATH, "PRESIDENTS", f"query{q_id}.json")
    if not os.path.exists(gt_file):
        return []
    try:
        with open(gt_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return data.get("result_set", data.get("tuples", []))
    except Exception as e:
        LOG.error(f"Error loading GT {gt_file}: {e}")
        return []


def load_results_from_folder(folder_path):
    results_map = {}
    if not os.path.exists(folder_path):
        return results_map

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isdir(file_path):
            continue

        if not filename.lower().endswith(".json") or not filename.lower().startswith("query"):
            continue

        match = re.search(r'query(\d+)\.json', filename)
        if not match:
            continue
        q_id = int(match.group(1))

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            res = data.get("result_set", data.get("tuples", []))
            results_map[q_id] = res
        except Exception as e:
            LOG.error(f"Error loading result {file_path}: {e}")
            continue

    return results_map


def exp_4():
    if not os.path.exists(GROUND_PATH):
        LOG.error("CRITICAL ERROR: Ground Truth folder not found!")
        return

    sql_map = load_queries_map(QUERIES_PATH)
    experiments = discover_result_folders()

    if not experiments:
        LOG.warning("No results found.")
        return

    all_metrics = []

    for method_name, folder_path in experiments:
        results_map = load_results_from_folder(folder_path)
        if not results_map:
            LOG.warning(f"No results for {method_name}")
            continue

        for q_id, sql in sql_map.items():
            if q_id not in results_map:
                continue

            gt_data = load_ground_truth(q_id)
            pred_data = results_map[q_id]

            avg, card = calculate_metrics_row(gt_data, pred_data)
            country, time_cat = categorize_query_id(q_id)

            all_metrics.append({
                'Method': method_name,
                'Country': country,
                'Time': time_cat,
                'AVG_SCORE': avg,
                'Cardinality': card
            })

    if not all_metrics:
        LOG.warning("No metrics calculated.")
        return

    df = pd.DataFrame(all_metrics)

    # TABLE 6 replication (grouped by Country + Time)
    print("\n" + "=" * 60)
    print("TABLE 6 REPLICATION")
    print("=" * 60)
    try:
        t6 = df.groupby(['Method', 'Country', 'Time'])['AVG_SCORE'].mean().unstack(level=0)
        # Optional reordering, depending on available methods
        print(t6)
    except Exception as e:
        print(f"Error producing Table 6: {e}")

    # TABLE 7 replication (grouped only by Time)
    print("\n" + "=" * 60)
    print("TABLE 7 REPLICATION")
    print("=" * 60)
    try:
        t7 = df.groupby(['Method', 'Time'])['AVG_SCORE'].mean().unstack(level=0)
        sorter = ['Recent', 'Past', 'All-Time']
        available = [s for s in sorter if s in t7.index]
        print(t7.reindex(available))
    except Exception as e:
        print(f"Error producing Table 7: {e}")


if __name__ == "__main__":
    exp_4()
