# src/galois/experiments/exp5_runner.py
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from sqlglot import exp, parse_one

# -----------------------------
# Query loading (reuse project layout: data/<DATASET>/queries_*.sql)
# -----------------------------
_SQL_SPLIT_RE = re.compile(r"(?im)^--\s*query[^\n]*\n")


def read_queries(dataset_dir: Path) -> List[str]:
    sql_files = sorted(dataset_dir.glob("queries_*.sql"))
    queries: List[str] = []
    for f in sql_files:
        text = f.read_text(encoding="utf-8")
        parts = _SQL_SPLIT_RE.split(text)
        for block in parts[1:]:
            q = block.strip()
            if not q:
                continue
            if q.endswith(";"):
                q = q[:-1].strip()
            if q:
                queries.append(q)
    return queries


# -----------------------------
# I/O: ground/submission table parsing
# Supports:
# - ground/<DATASET>/queryN.json  (list[dict] OR {"columns":..,"rows":..} OR {"result_set": ...})
# - submissions/<DATASET>/queryN.json (submission payloads with "result_set" plus tokens/time)
# -----------------------------
def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _norm(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (int, float)):
        # normalize numerics similar-ish to Java implementation: keep as string
        # (we apply numeric similarity later)
        return str(v)
    s = str(v).strip()
    return s


def _parse_table_obj(obj: Any) -> Tuple[List[str], List[List[Any]]]:
    # Accept several formats
    if isinstance(obj, dict):
        if "columns" in obj and "rows" in obj and isinstance(obj["rows"], list):
            cols = [str(c) for c in obj["columns"]]
            rows = [[_norm(x) for x in r] for r in obj["rows"]]
            return cols, rows

        if "result_set" in obj:
            return _parse_table_obj(obj["result_set"])

        # list-of-dicts in wrapper
        if "tuples" in obj and isinstance(obj["tuples"], list):
            arr = obj["tuples"]
            if arr and isinstance(arr[0], dict):
                cols = sorted({k for d in arr for k in d.keys()})
                rows = [[_norm(d.get(c)) for c in cols] for d in arr]
                return cols, rows
            return [], []

    if isinstance(obj, list):
        if not obj:
            return [], []
        if isinstance(obj[0], dict):
            cols = sorted({k for d in obj for k in d.keys()})
            rows = [[_norm(d.get(c)) for c in cols] for d in obj]
            return cols, rows
        if isinstance(obj[0], list):
            w = max(len(r) for r in obj)
            cols = [f"c{i}" for i in range(w)]
            rows = [[_norm(r[i]) if i < len(r) else "" for i in range(w)] for r in obj]
            return cols, rows

    return [], []


def read_ground_file(path: Path) -> Tuple[List[str], List[List[Any]]]:
    obj = _json_load(path)
    return _parse_table_obj(obj)


def read_submission_file(path: Optional[Path]) -> Tuple[List[str], List[List[Any]], Optional[float], Optional[float]]:
    if not path or not path.exists():
        return [], [], None, None
    obj = _json_load(path)
    cols, rows = _parse_table_obj(obj.get("result_set") if isinstance(obj, dict) else obj)
    tokens = None
    time_s = None
    if isinstance(obj, dict):
        try:
            tokens = float(obj.get("tokens")) if obj.get("tokens") is not None else None
        except Exception:
            tokens = None
        try:
            time_s = float(obj.get("time")) if obj.get("time") is not None else None
        except Exception:
            time_s = None
    return cols, rows, tokens, time_s


# -----------------------------
# Metrics (aligned with galois_eval defaults)
# - F1-Cell: exact or similarity (±10% for numeric, 10% edit distance for text)
# - Cardinality
# - Tuple constraint (exact multiplicity on normalized rows) OR tuple similarity constraint
# -----------------------------
_numeric_re = re.compile(r"^-?\d+(\.\d+)?$")


def _normalize_cell(s: str) -> str:
    s2 = s.replace(",", "").strip().lower().replace("\n", " ")
    return s2


def _try_float(s: str) -> Optional[float]:
    ss = s.strip()
    if not ss:
        return None
    # allow 1,23 locale by replacing comma with dot if no dot present
    if "," in ss and "." not in ss:
        ss = ss.replace(",", ".")
    try:
        return float(ss)
    except Exception:
        return None


def _edit_distance(a: str, b: str) -> int:
    # DP Levenshtein (small + safe)
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    if la > lb:
        a, b = b, a
        la, lb = lb, la
    prev = list(range(la + 1))
    for j in range(1, lb + 1):
        cj = [j] + [0] * la
        bj = b[j - 1]
        for i in range(1, la + 1):
            cj[i] = min(
                prev[i] + 1,
                cj[i - 1] + 1,
                prev[i - 1] + (0 if a[i - 1] == bj else 1),
            )
        prev = cj
    return prev[la]


def cells_similar(a: str, b: str) -> bool:
    # numeric ±10%
    fa = _try_float(a)
    fb = _try_float(b)
    if fa is not None and fb is not None:
        if fa == 0.0:
            return abs(fb) <= 0.1
        return abs((fa - fb) / fa) <= 0.1

    na = _normalize_cell(a)
    nb = _normalize_cell(b)
    if na == nb:
        return True

    k = int(math.floor(len(na) * 0.10))
    if abs(len(na) - len(nb)) > k:
        return False
    return _edit_distance(na, nb) <= k


def _cells_set(rows: List[List[Any]]) -> List[str]:
    out: List[str] = []
    for r in rows:
        for v in r:
            out.append(_normalize_cell(_norm(v)))
    # use list (not set) only for exactness? For F1-cell they use set semantics.
    return out


def f1_cell_exact(gt_rows: List[List[Any]], pr_rows: List[List[Any]]) -> float:
    gt = set(_cells_set(gt_rows))
    pr = set(_cells_set(pr_rows))
    if not gt and not pr:
        return 1.0
    if not gt or not pr:
        return 0.0
    inter = len(gt & pr)
    prec = inter / len(pr)
    rec = inter / len(gt)
    return 0.0 if (prec + rec) == 0 else (2 * prec * rec) / (prec + rec)


def f1_cell_similarity(gt_rows: List[List[Any]], pr_rows: List[List[Any]]) -> float:
    gt = list(set(_cells_set(gt_rows)))
    pr = list(set(_cells_set(pr_rows)))
    if not gt and not pr:
        return 1.0
    if not gt or not pr:
        return 0.0

    p_count = 0
    for rc in pr:
        if any(cells_similar(ec, rc) for ec in gt):
            p_count += 1
    precision = p_count / len(pr)

    r_count = 0
    for ec in gt:
        if any(cells_similar(ec, rc) for rc in pr):
            r_count += 1
    recall = r_count / len(gt)

    return 0.0 if (precision + recall) == 0 else (2 * precision * recall) / (precision + recall)


def cardinality(gt_rows: List[List[Any]], pr_rows: List[List[Any]]) -> float:
    se, sa = len(gt_rows), len(pr_rows)
    if se == 0 and sa == 0:
        return 1.0
    if se == 0 or sa == 0:
        return 0.0
    return min(se, sa) / max(se, sa)


def _norm_row(r: List[Any]) -> Tuple[str, ...]:
    return tuple(_normalize_cell(_norm(x)) for x in r)


def tuple_constraint(gt_rows: List[List[Any]], pr_rows: List[List[Any]]) -> float:
    # exact multiplicity match (Java-like)
    from collections import Counter

    gt_count = Counter(_norm_row(r) for r in gt_rows)
    pr_count = Counter(_norm_row(r) for r in pr_rows)
    if not gt_count and not pr_count:
        return 1.0
    if not gt_count:
        return 0.0

    good = 0
    for row, c in gt_count.items():
        good += 1 if pr_count.get(row) == c else 0
    return good / len(gt_count)


def tuple_similarity_constraint(gt_rows: List[List[Any]], pr_rows: List[List[Any]]) -> float:
    # row-level similarity with multiplicity bucket (conservative)
    from collections import Counter, defaultdict

    gt_count = Counter(_norm_row(r) for r in gt_rows)
    pr_count = Counter(_norm_row(r) for r in pr_rows)
    if not gt_count and not pr_count:
        return 1.0
    if not gt_count:
        return 0.0

    buckets: Dict[Tuple[int, int], List[Tuple[str, ...]]] = defaultdict(list)
    for prow, pcnt in pr_count.items():
        buckets[(len(prow), pcnt)].append(prow)

    def rows_similar(a: Tuple[str, ...], b: Tuple[str, ...]) -> bool:
        if len(a) != len(b):
            return False
        for i in range(len(a)):
            if a[i] == b[i]:
                continue
            if not cells_similar(a[i], b[i]):
                return False
        return True

    satisfied = 0
    for erow, ecnt in gt_count.items():
        candidates = buckets.get((len(erow), ecnt), [])
        if any(rows_similar(erow, prow) for prow in candidates):
            satisfied += 1
    return satisfied / len(gt_count)


# -----------------------------
# EXP-5 categorization: WHERE conditions (textual vs numerical)
# -----------------------------
@dataclass(frozen=True)
class Bucket:
    key: str
    label: str


BUCKETS: List[Bucket] = [
    Bucket("1_text", "1 TEXTUAL"),
    Bucket("gt1_text", ">1 TEXTUAL"),
    Bucket("gt1_text_1_num", ">1 TEXT., 1 NUMER."),
    Bucket("gt1_text_gt1_num", ">1 TEXT., >1 NUMER."),
    Bucket("1_num", "1 NUMERICAL"),
    Bucket("gt1_num", ">1 NUMERICAL"),
    Bucket("other", "OTHER"),
]


def _iter_predicates(where_expr: exp.Expression) -> Iterable[exp.Expression]:
    # We treat each Comparison/Like/In/Between as one predicate.
    for node in where_expr.walk():
        if isinstance(node, (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.Like, exp.ILike, exp.In, exp.Between)):
            yield node


def _predicate_type(pred: exp.Expression) -> Optional[str]:
    """
    Returns "text" or "num" if predicate clearly compares against a literal.
    Returns None if unknown (e.g., col-to-col).
    """
    # LIKE is textual by nature
    if isinstance(pred, (exp.Like, exp.ILike)):
        return "text"

    # BETWEEN: look at low/high
    if isinstance(pred, exp.Between):
        low = pred.args.get("low")
        high = pred.args.get("high")
        for lit in (low, high):
            if isinstance(lit, exp.Literal):
                if lit.is_string:
                    return "text"
                if _try_float(lit.this) is not None:
                    return "num"
        return None

    # IN: if any string literal -> text; else if numeric literals -> num
    if isinstance(pred, exp.In):
        exprs = pred.expressions or []
        seen_text = False
        seen_num = False
        for e in exprs:
            if isinstance(e, exp.Literal):
                if e.is_string:
                    seen_text = True
                elif _try_float(e.this) is not None:
                    seen_num = True
        if seen_text:
            return "text"
        if seen_num:
            return "num"
        return None

    # Comparisons: check sides for Literal
    left = pred.left
    right = pred.right
    for side in (left, right):
        if isinstance(side, exp.Literal):
            if side.is_string:
                return "text"
            if _try_float(side.this) is not None:
                return "num"
    return None


def bucket_for_sql(sql: str) -> Tuple[str, int, int]:
    """
    Returns (bucket_key, text_count, num_count).
    """
    try:
        parsed = parse_one(sql, read="duckdb")
    except Exception:
        return "other", 0, 0

    where = parsed.args.get("where")
    if not isinstance(where, exp.Where) or where.this is None:
        return "other", 0, 0

    text_cnt = 0
    num_cnt = 0
    unknown = 0
    for pred in _iter_predicates(where.this):
        t = _predicate_type(pred)
        if t == "text":
            text_cnt += 1
        elif t == "num":
            num_cnt += 1
        else:
            unknown += 1

    # Paper buckets (Table 8 style)
    if text_cnt == 1 and num_cnt == 0:
        return "1_text", text_cnt, num_cnt
    if text_cnt > 1 and num_cnt == 0:
        return "gt1_text", text_cnt, num_cnt
    if text_cnt > 1 and num_cnt == 1:
        return "gt1_text_1_num", text_cnt, num_cnt
    if text_cnt > 1 and num_cnt > 1:
        return "gt1_text_gt1_num", text_cnt, num_cnt
    if text_cnt == 0 and num_cnt == 1:
        return "1_num", text_cnt, num_cnt
    if text_cnt == 0 and num_cnt > 1:
        return "gt1_num", text_cnt, num_cnt

    return "other", text_cnt, num_cnt


# -----------------------------
# Dataset folder resolution (handles GEO vs geo vs flight-2 vs FLIGHT-2)
# -----------------------------
def resolve_dataset_dir(root: Path, dataset: str) -> Optional[Path]:
    cand = [
        dataset,
        dataset.lower(),
        dataset.upper(),
        dataset.replace("_", "-"),
        dataset.replace("_", "-").lower(),
        dataset.replace("_", "-").upper(),
    ]
    for c in cand:
        p = root / c
        if p.exists() and p.is_dir():
            return p
    # try case-insensitive search
    if root.exists():
        for p in root.iterdir():
            if p.is_dir() and p.name.lower() == dataset.lower():
                return p
    return None


# -----------------------------
# Table printing (clean ASCII box)
# -----------------------------
def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _fmt_m(x: Optional[float]) -> str:
    if x is None:
        return ""
    return f"{(x / 1_000_000):.3f}"


def print_box_table(headers: List[str], rows: List[List[str]]) -> None:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def line(ch: str = "-") -> str:
        return "+" + "+".join(ch * (w + 2) for w in widths) + "+"

    def row(cells: List[str]) -> str:
        return "| " + " | ".join(cells[i].ljust(widths[i]) for i in range(len(headers))) + " |"

    print(line("-"))
    print(row(headers))
    print(line("="))
    for r in rows:
        print(row(r))
        print(line("-"))


# -----------------------------
# Main EXP-5
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Exp-5 (Table 8 style): group queries by WHERE condition types and report AVG-Score.")
    ap.add_argument("--ground", required=True, type=Path, help="Ground truth root (contains per-dataset folders with queryN.json).")
    ap.add_argument("--data", required=True, type=Path, help="SQL dataset root (contains data/<dataset>/queries_*.sql).")
    ap.add_argument(
        "--submissions",
        required=True,
        nargs="+",
        type=Path,
        help="Submission roots in order: NL SQL GALOIS_WO GALOIS_A GALOIS_F (5 paths).",
    )
    ap.add_argument("--datasets", nargs="*", default=None, help="Datasets to include (default: all under --ground).")
    ap.add_argument("--cell-metric", choices=["exact", "similarity"], default="similarity")
    ap.add_argument("--tuple-metric", choices=["constraint", "similarity"], default="constraint")
    ap.add_argument("--format", choices=["table", "csv", "json"], default="table")
    ap.add_argument("--plot-out", default=None, help="If set, saves a PNG with AVG-Score bars per bucket.")
    args = ap.parse_args()

    if len(args.submissions) != 5:
        print("ERROR: --submissions must contain exactly 5 paths: NL SQL GALOIS_WO GALOIS_A GALOIS_F", file=sys.stderr)
        sys.exit(2)

    sub_names = ["NL", "SQL", "GALOIS_WO", "GALOIS_A", "GALOIS_F"]
    sub_roots = dict(zip(sub_names, args.submissions))

    # choose metric fns
    f1_fn = f1_cell_similarity if args.cell_metric == "similarity" else f1_cell_exact
    tcon_fn = tuple_similarity_constraint if args.tuple_metric == "similarity" else tuple_constraint

    # datasets
    if args.datasets:
        datasets = args.datasets
    else:
        datasets = sorted([p.name for p in args.ground.iterdir() if p.is_dir() and not p.name.startswith(".")])

    # accumulators: bucket -> system -> list[avg_score]
    scores: Dict[str, Dict[str, List[float]]] = {b.key: {s: [] for s in sub_names} for b in BUCKETS}
    tokens_sum: Dict[str, Dict[str, float]] = {b.key: {s: 0.0 for s in sub_names} for b in BUCKETS}
    times_sum: Dict[str, Dict[str, float]] = {b.key: {s: 0.0 for s in sub_names} for b in BUCKETS}
    counts: Dict[str, int] = {b.key: 0 for b in BUCKETS}

    # iterate datasets + queries
    for ds in datasets:
        gt_dir = resolve_dataset_dir(args.ground, ds)
        if gt_dir is None:
            continue

        data_dir = resolve_dataset_dir(args.data, ds)
        if data_dir is None:
            # if SQL files are stored as data/<dataset>, we still need them for bucketing
            # skip if missing
            continue

        sql_queries = read_queries(data_dir)
        if not sql_queries:
            continue

        # ground files are query1.json ...
        for i, sql in enumerate(sql_queries, start=1):
            gt_path = gt_dir / f"query{i}.json"
            if not gt_path.exists():
                # keep alignment conservative: skip missing
                continue

            bucket_key, _, _ = bucket_for_sql(sql)
            if bucket_key not in scores:
                bucket_key = "other"
            counts[bucket_key] += 1

            gt_cols, gt_rows = read_ground_file(gt_path)

            for sys_name in sub_names:
                sub_root = sub_roots[sys_name]
                sub_ds_dir = resolve_dataset_dir(sub_root, ds)
                sub_path = (sub_ds_dir / f"query{i}.json") if sub_ds_dir else None

                pr_cols, pr_rows, q_tokens, q_time = read_submission_file(sub_path)
                # align predicted columns to GT columns for tuple metrics if needed
                # (simple projection by GT column order if prediction is dict-like got reshaped)
                # Our table parser already returns rows aligned to cols; GT is independent.
                f1 = f1_fn(gt_rows, pr_rows)
                card = cardinality(gt_rows, pr_rows)
                tcon = tcon_fn(gt_rows, pr_rows)
                avg = mean([f1, card, tcon])

                scores[bucket_key][sys_name].append(avg)
                if q_tokens is not None:
                    tokens_sum[bucket_key][sys_name] += float(q_tokens)
                if q_time is not None:
                    times_sum[bucket_key][sys_name] += float(q_time)

    # build output rows in paper order
    out_rows: List[List[str]] = []
    json_out: Dict[str, Any] = {}

    headers = ["Condition(s)", "#Q"] + sub_names

    for b in BUCKETS:
        n = counts[b.key]
        if n == 0:
            # keep rows with 0 queries (paper-style tables usually omit; we keep for transparency)
            avgs = ["0.000"] * len(sub_names)
        else:
            avgs = []
            for sys_name in sub_names:
                vals = scores[b.key][sys_name]
                avgs.append(_fmt(sum(vals) / len(vals)) if vals else "0.000")

        out_rows.append([b.label, str(n)] + avgs)
        json_out[b.label] = {"#Q": n, **{sub_names[j]: float(avgs[j]) for j in range(len(sub_names))}}

    # print
    if args.format == "json":
        print(json.dumps(json_out, indent=2, ensure_ascii=False))
    elif args.format == "csv":
        w = csv.writer(sys.stdout)
        w.writerow(headers)
        for r in out_rows:
            w.writerow(r)
    else:
        print_box_table(headers, out_rows)

    # plot output when requested
    if args.plot_out:
        # grouped bar chart: buckets on x, 5 series
        labels = [b.label for b in BUCKETS]
        x = list(range(len(labels)))
        width = 0.15

        fig, ax = plt.subplots()
        for j, sys_name in enumerate(sub_names):
            y = []
            for b in BUCKETS:
                vals = scores[b.key][sys_name]
                y.append((sum(vals) / len(vals)) if vals else 0.0)
            ax.bar([xi + (j - 2) * width for xi in x], y, width=width, label=sys_name)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("AVG-Score")
        ax.set_title("Exp-5: AVG-Score by WHERE condition type/count")
        ax.legend()
        fig.tight_layout()
        Path(args.plot_out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.plot_out, dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
