from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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


SKIP_BASELINE_METHODS = {
    "__pycache__",
    "_tmp",
    "archive",
    "tmp",
}


def _read_queries(sql_path: Path) -> List[str]:
    lines = sql_path.read_text(encoding="utf-8").splitlines()
    cleaned = [line for line in lines if not line.strip().startswith("--")]
    joined = "\n".join(cleaned)
    queries = [q.strip() for q in joined.split(";") if q.strip()]
    return queries


def categorize_query_id(q_id: int) -> Tuple[str, str]:
    vz_recent = {8, 10, 13}
    vz_past = {11, 12}
    vz_all = {1, 2, 3, 4, 5, 6, 7, 9}

    usa_recent = {18, 21, 24, 26}
    usa_past = {23, 25}
    usa_all = {14, 15, 16, 17, 19, 20, 22}

    if q_id in vz_recent:
        return "Venezuela", "Recent"
    if q_id in vz_past:
        return "Venezuela", "Past"
    if q_id in vz_all:
        return "Venezuela", "All-Time"
    if q_id in usa_recent:
        return "USA", "Recent"
    if q_id in usa_past:
        return "USA", "Past"
    if q_id in usa_all:
        return "USA", "All-Time"
    raise ValueError(f"Uncategorized query id: {q_id}")


def _normalize_key(key: str) -> str:
    return key.split(".")[-1].lower()


def _normalize_row_keys(row: Dict[str, Any]) -> Dict[str, Any]:
    return {_normalize_key(k): v for k, v in row.items()}


def _normalize_table_obj(obj: Any) -> Any:
    if isinstance(obj, dict) and "rows" in obj:
        rows = obj.get("rows")
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            new_obj = dict(obj)
            new_obj["rows"] = [_normalize_row_keys(r) for r in rows]
            return new_obj
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        return [_normalize_row_keys(r) for r in obj]
    return obj


def _extract_table_obj(obj: Any) -> Any:
    if isinstance(obj, dict) and "result_set" in obj:
        obj = obj["result_set"]
    if isinstance(obj, dict) and "tuples" in obj:
        return obj["tuples"]
    if isinstance(obj, list):
        return obj
    return obj


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_table(path: Path) -> Tuple[List[str], List[List[Any]]]:
    if not path.exists():
        return [], []
    obj = _load_json(path)
    table_obj = _extract_table_obj(obj)
    table_obj = _normalize_table_obj(table_obj)
    return _parse_table_from_obj(table_obj)


def _collect_galois_methods() -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    root = DATA_DIR / ".galois_output"
    if not root.exists():
        return out
    for method_dir in root.iterdir():
        if not method_dir.is_dir():
            continue
        ds_dir = method_dir / "PRESIDENTS"
        if ds_dir.exists():
            out[method_dir.name] = ds_dir
    return out


def _collect_baseline_methods() -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    root = DATA_DIR / ".output"
    if not root.exists():
        return out
    for method_dir in root.iterdir():
        if not method_dir.is_dir():
            continue
        name = method_dir.name
        if name.startswith(".") or name in SKIP_BASELINE_METHODS:
            continue
        ds_dir = method_dir / "watsonx" / "PRESIDENTS"
        if ds_dir.exists():
            out[name] = ds_dir
    return out


def _print_table_6_replication(rows: List[Dict[str, Any]]) -> None:
    print("TABLE 6 REPLICATION")
    if not rows:
        print("No results found.")
        return

    stats: Dict[Tuple[str, str], List[float]] = {}
    methods = sorted({r["Method"] for r in rows})
    countries_seen = {r["Country"] for r in rows}
    country_order = ["Venezuela", "USA"]
    countries = [c for c in country_order if c in countries_seen]
    countries.extend(sorted(c for c in countries_seen if c not in countries))

    for r in rows:
        key = (r["Method"], r["Country"])
        stats.setdefault(key, []).append(r["AVG_SCORE"])

    headers = ["Method"] + countries
    table_rows: List[List[str]] = []
    for method in methods:
        row = [method]
        for country in countries:
            values = stats.get((method, country), [])
            mean_val = sum(values) / len(values) if values else None
            row.append(f"{mean_val:.3f}" if mean_val is not None else "")
        table_rows.append(row)

    widths = [len(h) for h in headers]
    for r in table_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def line() -> None:
        print("+" + "+".join("-" * (w + 2) for w in widths) + "+")

    def row(cells: Iterable[str]) -> None:
        print("| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |")

    line()
    row(headers)
    line()
    for r in table_rows:
        row(r)
    line()


def main() -> None:
    sql_path = DATA_DIR / "PRESIDENTS" / "queries_presidents.sql"
    queries = _read_queries(sql_path)
    q_ids = list(range(1, len(queries) + 1))

    methods: Dict[str, Path] = {}
    methods.update(_collect_galois_methods())
    methods.update(_collect_baseline_methods())

    rows: List[Dict[str, Any]] = []
    for method, method_dir in sorted(methods.items()):
        for q_id in q_ids:
            gt_path = GROUND_PATH / "PRESIDENTS" / f"query{q_id}.json"
            pr_path = method_dir / f"query{q_id}.json"
            if not gt_path.exists():
                continue

            gt_cols, gt_rows = _load_table(gt_path)
            pr_cols, pr_rows = _load_table(pr_path)

            f1 = f1_cell_similarity(gt_cols, gt_rows, pr_cols, pr_rows)
            card = cardinality(gt_rows, pr_rows)
            tcon = tuple_constraint(gt_rows, pr_rows)
            avg = (f1 + card + tcon) / 3.0

            country, time_label = categorize_query_id(q_id)
            rows.append(
                {
                    "Method": method,
                    "Country": country,
                    "Time": time_label,
                    "AVG_SCORE": avg,
                    "Cardinality": card,
                }
            )

    _print_table_6_replication(rows)


if __name__ == "__main__":
    main()
