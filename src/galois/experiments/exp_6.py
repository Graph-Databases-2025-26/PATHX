from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt

def print_pretty_table(headers: List[str], rows: List[List[str]]) -> None:
    widths = [len(h) for h in headers]
    for r in rows:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def line(ch: str) -> str:
        return "+" + "+".join(ch * (w + 2) for w in widths) + "+"

    def row(cells: List[str]) -> str:
        return "| " + " | ".join(cells[i].ljust(widths[i]) for i in range(len(headers))) + " |"

    print(line("-"))
    print(row(headers))
    print(line("="))
    for r in rows:
        print(row(r))
        print(line("-"))


# -------------------------
# Stats extraction helpers
# -------------------------
def _to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return int(float(s))
        except Exception:
            return None
    return None


def _sum_int_list(x: Any) -> Optional[int]:
    if not isinstance(x, list):
        return None
    vals: List[int] = []
    for v in x:
        iv = _to_int(v)
        if iv is None:
            return None
        vals.append(iv)
    return sum(vals)


def extract_iters(meta: Dict[str, Any]) -> int:
    # common keys seen in projects
    for k in ("iterations", "num_iterations", "iters", "n_iters"):
        v = _to_int(meta.get(k))
        if v is not None and v > 0:
            return v

    # sometimes scans store an "iter_logs" list
    if isinstance(meta.get("iter_logs"), list) and len(meta["iter_logs"]) > 0:
        return len(meta["iter_logs"])

    return 1


def extract_input_tokens_all_iters(obj: Dict[str, Any]) -> Optional[int]:
    """
    Prefer explicit per-iteration prompt/input tokens.
    Falls back to total tokens (obj["tokens"]) if nothing else exists.
    """
    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}

    # strongest signals: per-iteration input/prompt tokens
    for k in (
        "input_tokens_per_iter",
        "prompt_tokens_per_iter",
        "iteration_input_tokens",
        "iteration_prompt_tokens",
        "iter_input_tokens",
        "iter_prompt_tokens",
    ):
        s = _sum_int_list(meta.get(k))
        if s is not None:
            return s

    # sometimes it’s stored as list of dict logs
    if isinstance(meta.get("iter_logs"), list):
        total = 0
        ok = False
        for it in meta["iter_logs"]:
            if not isinstance(it, dict):
                return None
            # allow multiple field names
            v = _to_int(it.get("input_tokens"))
            if v is None:
                v = _to_int(it.get("prompt_tokens"))
            if v is None:
                # if one log lacks it, bail out to avoid mixing partials
                return None
            total += v
            ok = True
        if ok:
            return total

    # weaker: tokens may already be "total tokens" (input+output) — but we use it if no better data
    t = _to_int(obj.get("tokens"))
    return t


@dataclass(frozen=True)
class QueryStats:
    input_tokens_all_iters: int
    iters: int


def load_query_stats(json_path: Path) -> Optional[QueryStats]:
    try:
        obj = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(obj, dict):
        return None

    meta = obj.get("meta") if isinstance(obj.get("meta"), dict) else {}
    iters = extract_iters(meta)

    inp = extract_input_tokens_all_iters(obj)
    if inp is None:
        return None

    return QueryStats(input_tokens_all_iters=inp, iters=iters)


def iter_query_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    files = sorted(folder.glob("**/query*.json"))
    return [p for p in files if p.is_file()]


# -------------------------
# Exp-6 runner
# -------------------------
def compute_exp6_stats(results_dir: Path) -> Tuple[float, float, int, int]:
    """
    Returns:
      avg_input_tokens_all_iters,
      avg_iters_per_query,
      n_queries_10plus_iters,
      n_queries_total
    """
    files = iter_query_files(results_dir)
    stats: List[QueryStats] = []

    for p in files:
        st = load_query_stats(p)
        if st is not None:
            stats.append(st)

    if not stats:
        return 0.0, 0.0, 0, 0

    avg_input = mean(s.input_tokens_all_iters for s in stats)
    avg_iters = mean(s.iters for s in stats)
    n_10plus = sum(1 for s in stats if s.iters >= 10)
    return avg_input, avg_iters, n_10plus, len(stats)


def plot_bar(title: str, labels: List[str], values: List[float], ylabel: str) -> None:
    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Variant")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def main() -> None:
    ap = argparse.ArgumentParser(description="Exp-6: Evolution of Prompt Size (Table 9 + plots).")
    ap.add_argument(
        "--variants",
        nargs="*",
        default=["results_galois_wo", "results_galois_a", "results_galois_f"],
        help="Folders with JSON results (default: results_galois_wo results_galois_a results_galois_f)",
    )
    args = ap.parse_args()

    # map folder name -> pretty label
    label_map = {
        "results_galois_wo": "GALOIS O",
        "results_galois_a": "GALOIS A",
        "results_galois_f": "GALOIS F",
        "galois_wo": "GALOIS O",
        "galois_a": "GALOIS A",
        "galois_f": "GALOIS F",
    }

    rows: List[List[str]] = []
    plot_labels: List[str] = []
    plot_input_tokens: List[float] = []
    plot_avg_iters: List[float] = []

    # table headers exactly like Table 9 spirit
    headers = ["Metric"] + [label_map.get(v.lower(), v) for v in args.variants]

    # collect stats per variant
    per_variant: List[Tuple[float, float, int, int]] = []
    pretty_names: List[str] = []

    for v in args.variants:
        p = Path(v)
        avg_input, avg_iters, n_10plus, n_total = compute_exp6_stats(p)
        per_variant.append((avg_input, avg_iters, n_10plus, n_total))
        pretty = label_map.get(v.lower(), v)
        pretty_names.append(pretty)

        plot_labels.append(pretty)
        plot_input_tokens.append(avg_input)
        plot_avg_iters.append(avg_iters)

    # build rows
    rows.append(["Avg # Input Tokens All Iters"] + [f"{x[0]:.2f}" for x in per_variant])
    rows.append(["Avg # Iters per Query"] + [f"{x[1]:.2f}" for x in per_variant])
    rows.append(["# Queries with 10+ Iters"] + [str(x[2]) for x in per_variant])

    print_pretty_table(headers, rows)

    # plots (required)
    plot_bar(
        "Exp-6 Prompt Size — Avg # Input Tokens (All Iterations)",
        plot_labels,
        plot_input_tokens,
        "Tokens",
    )
    plot_bar(
        "Exp-6 Prompt Size — Avg # Iterations per Query",
        plot_labels,
        plot_avg_iters,
        "Iterations",
    )

    # small diagnostic note (useful in practice)
    for name, (_, _, _, n_total) in zip(pretty_names, per_variant):
        if n_total == 0:
            print(f"[WARN] {name}: no query*.json found (or none parsable). Check folder path.")


if __name__ == "__main__":
    main()
