import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

from src.config import Config_Loader
from src.db import load_queries_from_folder


def retry_call(fn, *, tries: int = 4, base_delay: float = 2.0, label: str = "call"):
    """Retry on common httpx errors with exponential backoff."""
    last_exc = None
    for attempt in range(tries):
        try:
            return fn()
        except (
            httpx.RemoteProtocolError,
            httpx.ReadTimeout,
            httpx.ConnectTimeout,
            httpx.ConnectError,
        ) as e:
            last_exc = e
            wait_s = base_delay * (2 ** attempt)
            LOG.warning(
                "[EXP2] %s failed (%s). Retry %d/%d in %.1fs",
                label,
                type(e).__name__,
                attempt + 1,
                tries,
                wait_s,
            )
            time.sleep(wait_s)

    LOG.error("[EXP2] %s failed after %d retries: %s", label, tries, last_exc)
    raise last_exc


def pick_existing_subdir(base: Path, candidates: List[str]) -> str:
    """Return first existing directory name under base, or the first candidate."""
    for name in candidates:
        if (base / name).exists():
            return name
    return candidates[0]


def _as_str(v: Any) -> str:
    return "" if v is None else str(v)


def load_reference_table(
    gt_root: Path,
    dataset_dir: str,
    query_id: str,
) -> Tuple[List[str], List[List[Any]]]:
    """Read ground truth for a query (simple JSON formats)."""
    p = gt_root / dataset_dir / f"{query_id}.json"
    if not p.exists():
        raise FileNotFoundError(f"Missing ground truth file: {p}")

    data = json.loads(p.read_text(encoding="utf-8"))

    # list of dict rows
    if isinstance(data, list) and (not data or isinstance(data[0], dict)):
        if not data:
            return [], []
        cols = list(data[0].keys())
        rows = [[_as_str(row.get(c)) for c in cols] for row in data]
        return cols, rows

    # {"columns": [...], "rows": [...]}
    if isinstance(data, dict) and "columns" in data and "rows" in data:
        cols = list(data["columns"])
        rows = [[_as_str(v) for v in r] for r in data["rows"]]
        return cols, rows

    raise ValueError(f"Unexpected ground-truth format in {p}")


def prediction_to_table(
    records: List[Dict[str, Any]],
    gt_cols: Optional[List[str]] = None,
) -> Tuple[List[str], List[List[Any]]]:
    """Turn a list of dicts into (cols, rows), aligned with gt_cols if given."""
    if not records:
        return (gt_cols or []), []

    if gt_cols:
        cols = list(gt_cols)
        rows = [[_as_str(r.get(c)) for c in cols] for r in records]
        return cols, rows

    cols = list(records[0].keys())
    rows = [[_as_str(r.get(c)) for c in cols] for r in records]
    return cols, rows


def eval_table(
    gt_cols: List[str],
    gt_rows: List[List[Any]],
    pr_cols: List[str],
    pr_rows: List[List[Any]],
) -> Dict[str, float]:
    """Compute the three metrics and their mean."""
    f1 = f1_cell_exact(gt_cols, gt_rows, pr_cols, pr_rows)
    card = cardinality(gt_rows, pr_rows)
    tcon = tuple_constraint(gt_rows, pr_rows)
    avg = (f1 + card + tcon) / 3.0
    return {"f1": f1, "cardinality": card, "tuple_constraint": tcon, "avg": avg}


def close_enough(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) <= eps


def select_pushdown_conditions(
    config,
    dataset_db: str,
    sql_query: str,
) -> List[str]:
    """Use the confidence estimator to pick WHERE conditions to push."""
    g = Galois(config, dataset_db, sql_query, physical_strategy="table")
    all_conds = g.parsed_sql.get("where_conditions", []) or []
    if not all_conds:
        return []

    estimator = ConfidenceEstimator(
        g.llm_wrapper,
        g.dataset,
        system_prompt_galois_confidence(),
        human_prompt_galois_confidence("CONDITION"),
    )

    confident = estimator.estimate_confidence_conditions(
        g.parsed_sql["from_table"], all_conds
    )

    if len(confident) == 0:
        return []
    if len(confident) == 1:
        return confident
    return all_conds


def run_plan_once(
    config,
    dataset_db: str,
    sql_query: str,
    pushed_conditions: List[str],
    physical: str,
    tag: str,
):
    """Build and execute a plan with a fixed physical strategy."""
    g = Galois(config, dataset_db, sql_query, physical_strategy=physical)
    plan = g.build_execution_plan(conditions_to_push=pushed_conditions)

    def do_call():
        return g.execute_variant(plan, tag)

    return retry_call(do_call, label=f"{physical}:{tag}")


def run_exp2(
    dataset_db: str,
    dataset_gt: str,
    provider: str,
    limit: Optional[int],
    only: str,
) -> None:
    config = Config_Loader().get_config()

    queries = load_queries_from_folder(DATA_DIR / dataset_db)
    if limit:
        queries = queries[:limit]

    phys_table_out: List[Dict[str, Any]] = []
    phys_key_out: List[Dict[str, Any]] = []
    phys_auto_out: List[Dict[str, Any]] = []

    phys_scores_table: List[float] = []
    phys_scores_key: List[float] = []
    phys_scores_auto: List[float] = []

    log_nopush_out: List[Dict[str, Any]] = []
    log_all_out: List[Dict[str, Any]] = []
    log_selected_out: List[Dict[str, Any]] = []

    log_scores_nopush: List[float] = []
    log_scores_all: List[float] = []
    log_scores_selected: List[float] = []

    auto_matches_best = 0
    auto_total = 0
    logical_total = 0

    for i, (_, sql_query) in enumerate(queries):
        idx = i + 1
        query_id = f"query{idx}"
        save_id = str(idx)

        LOG.info("[EXP2] Running %s", query_id)

        try:
            gt_cols, gt_rows = load_reference_table(GROUND_PATH, dataset_gt, query_id)
        except Exception as e:
            LOG.warning("[EXP2] Skipping %s (ground truth error: %s)", query_id, e)
            continue

        try:
            selected_conds = select_pushdown_conditions(config, dataset_db, sql_query)
        except Exception as e:
            LOG.warning("[EXP2] Could not compute pushdown conditions for %s: %s",
                        query_id, e)
            selected_conds = []

        # table / key / auto with the same pushed conditions
        if only in ("physical", "all"):
            try:
                r_t, s_t = run_plan_once(
                    config,
                    dataset_db,
                    sql_query,
                    pushed_conditions=selected_conds,
                    physical="table",
                    tag="PHYS_TABLE",
                )
                c_t, rows_t = prediction_to_table(r_t, gt_cols)
                m_t = eval_table(gt_cols, gt_rows, c_t, rows_t)
                phys_table_out.append(
                    {
                        "query_id": save_id,
                        "result_set": {"columns": c_t, "rows": rows_t},
                        "total_tokens": s_t.get("total_tokens", 0),
                        "total_time": s_t.get("total_time", 0.0),
                    }
                )
                phys_scores_table.append(m_t["avg"])

                r_k, s_k = run_plan_once(
                    config,
                    dataset_db,
                    sql_query,
                    pushed_conditions=selected_conds,
                    physical="key",
                    tag="PHYS_KEY",
                )
                c_k, rows_k = prediction_to_table(r_k, gt_cols)
                m_k = eval_table(gt_cols, gt_rows, c_k, rows_k)
                phys_key_out.append(
                    {
                        "query_id": save_id,
                        "result_set": {"columns": c_k, "rows": rows_k},
                        "total_tokens": s_k.get("total_tokens", 0),
                        "total_time": s_k.get("total_time", 0.0),
                    }
                )
                phys_scores_key.append(m_k["avg"])

                r_a, s_a = run_plan_once(
                    config,
                    dataset_db,
                    sql_query,
                    pushed_conditions=selected_conds,
                    physical="auto",
                    tag="PHYS_AUTO",
                )
                c_a, rows_a = prediction_to_table(r_a, gt_cols)
                m_a = eval_table(gt_cols, gt_rows, c_a, rows_a)
                phys_auto_out.append(
                    {
                        "query_id": save_id,
                        "result_set": {"columns": c_a, "rows": rows_a},
                        "total_tokens": s_a.get("total_tokens", 0),
                        "total_time": s_a.get("total_time", 0.0),
                    }
                )
                phys_scores_auto.append(m_a["avg"])

                best_fixed = max(m_t["avg"], m_k["avg"])
                auto_total += 1
                if close_enough(m_a["avg"], best_fixed):
                    auto_matches_best += 1

            except Exception as e:
                LOG.error("[EXP2] Physical part failed for %s: %s", query_id, e)

        # comparison of pushdown styles on table scan
        if only in ("logical", "all"):
            try:
                g_tmp = Galois(config, dataset_db, sql_query, physical_strategy="table")
                all_conditions = g_tmp.parsed_sql.get("where_conditions", []) or []
            except Exception as e:
                LOG.warning("[EXP2] Cannot inspect WHERE clause for %s: %s",
                            query_id, e)
                all_conditions = []

            if len(all_conditions) < 2:
                LOG.info("[EXP2] Skipping logical comparison for %s (less than 2 conditions)",
                         query_id)
                continue

            logical_total += 1

            variants = [
                ("LOG_NOPUSH", [], log_nopush_out, log_scores_nopush),
                ("LOG_ALL", all_conditions, log_all_out, log_scores_all),
                ("LOG_SELECTED", selected_conds, log_selected_out, log_scores_selected),
            ]

            for tag, conds, bucket_out, bucket_scores in variants:
                try:
                    r_l, s_l = run_plan_once(
                        config,
                        dataset_db,
                        sql_query,
                        pushed_conditions=conds,
                        physical="table",
                        tag=tag,
                    )
                except Exception as e:
                    LOG.error("[EXP2] Logical variant %s failed for %s: %s",
                              tag, query_id, e)
                    continue

                c_l, rows_l = prediction_to_table(r_l, gt_cols)
                m_l = eval_table(gt_cols, gt_rows, c_l, rows_l)

                bucket_out.append(
                    {
                        "query_id": save_id,
                        "result_set": {"columns": c_l, "rows": rows_l},
                        "total_tokens": s_l.get("total_tokens", 0),
                        "total_time": s_l.get("total_time", 0.0),
                    }
                )
                bucket_scores.append(m_l["avg"])

    if only in ("physical", "all"):
        save_galois_results(phys_table_out, "PHYS_TABLE", provider, dataset_db)
        save_galois_results(phys_key_out, "PHYS_KEY", provider, dataset_db)
        save_galois_results(phys_auto_out, "PHYS_AUTO", provider, dataset_db)

        if auto_total > 0:
            acc = auto_matches_best / auto_total
            LOG.info(
                "[EXP2] Auto physical strategy matched the best fixed plan in %.1f%% of queries",
                acc * 100.0,
            )

    if only in ("logical", "all"):
        save_galois_results(log_nopush_out, "LOG_NOPUSH_TABLE", provider, dataset_db)
        save_galois_results(log_all_out, "LOG_ALL_TABLE", provider, dataset_db)
        save_galois_results(log_selected_out, "LOG_SELECTED_TABLE", provider, dataset_db)

        LOG.info("[EXP2] Logical comparison done on %d queries", logical_total)

        def _avg(xs: List[float]) -> float:
            return float(sum(xs) / len(xs)) if xs else 0.0

        def _tokens_m(results: List[dict]) -> float:
            return float(sum(r.get("total_tokens", 0) for r in results)) / 1e6

        avg_np = _avg(log_scores_nopush)
        avg_all = _avg(log_scores_all)
        avg_sel = _avg(log_scores_selected)

        tok_np_m = _tokens_m(log_nopush_out)
        tok_all_m = _tokens_m(log_all_out)
        tok_sel_m = _tokens_m(log_selected_out)

        LOG.info(
            "[EXP2] Logical AVG-score  | NO_PUSH=%.3f  ALL=%.3f  SELECTED=%.3f",
            avg_np,
            avg_all,
            avg_sel,
        )
        LOG.info(
            "[EXP2] Logical tokens (M) | NO_PUSH=%.3f  ALL=%.3f  SELECTED=%.3f",
            tok_np_m,
            tok_all_m,
            tok_sel_m,
        )

        print("\nLogical summary (AVG-score / tokens in M):")
        print(
            f"NO-PUSH  : avg={avg_np:.3f}, tokens={tok_np_m:.3f}\n"
            f"GALOIS A : avg={avg_all:.3f}, tokens={tok_all_m:.3f}\n"
            f"GALOIS F : avg={avg_sel:.3f}, tokens={tok_sel_m:.3f}"
        )

        print("\nLaTeX snippet:\n")
        print(r"\begin{tabular}{lccc}")
        print(r"\hline")
        print(r"Metric        & NO-PUSH & GALOIS A & GALOIS F \\")
        print(r"\hline")
        print(
            f"AVG-SCORE     & {avg_np:.3f} & {avg_all:.3f} & {avg_sel:.3f} \\\\"
        )
        print(
            f"# TOKENS in M & {tok_np_m:.3f} & {tok_all_m:.3f} & {tok_sel_m:.3f} \\\\"
        )
        print(r"\hline")
        print(r"\end{tabular}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--ground-dataset", default=None)
    parser.add_argument("--provider", default="watsonx")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", choices=["physical", "logical", "all"], default="all")
    args = parser.parse_args()

    dataset_db = args.dataset or pick_existing_subdir(DATA_DIR, ["geo", "GEO"])
    dataset_gt = args.ground_dataset or pick_existing_subdir(GROUND_PATH, ["GEO", "geo"])

    run_exp2(
        dataset_db=dataset_db,
        dataset_gt=dataset_gt,
        provider=args.provider,
        limit=args.limit,
        only=args.only,
    )


if __name__ == "__main__":
    main()
