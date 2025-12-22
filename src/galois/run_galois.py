# src/galois/run_galois.py
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import duckdb
import pandas as pd
from sqlglot import exp, parse_one

from ..config.config import load_config, AppConfig  # type: ignore
from ..utils.dataset_io import discover_datasets, read_queries  # type: ignore
from ..baselines.llm_factory import get_llm_wrapper  # type: ignore

from .variants.galois_wo import GaloisWOOperator
from .variants.galois_a import GaloisA
from .variants.galois_s import GaloisS
from .variants.galois_f import GaloisF


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ----------------------------
# SQL parsing + column planning (single-table focus)
# ----------------------------
@dataclass(frozen=True)
class ParsedSQL:
    sql: str
    table: str
    table_aliases: Set[str]
    where_clause: Optional[str]
    has_aggregate_or_grouping: bool
    select_plain_columns: Optional[List[str]]  # None if SELECT has expressions/aggregates
    scan_columns: List[str]  # what we ask the LLM to return


def _dataset_duckdb_path(dataset: str) -> Path:
    """
    Best-effort resolution of the dataset .duckdb file.
    - Common: db/<dataset>.duckdb
    - Fallback: first *.duckdb found under db/
    """
    ds_dir = Path("db") 
    preferred = ds_dir / f"{dataset}.duckdb"
    if preferred.exists():
        return preferred

    candidates = sorted(ds_dir.glob("*.duckdb"))
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"No .duckdb file found under: {ds_dir}")


def _duckdb_table_columns(dataset: str, table: str) -> List[str]:
    db_path = _dataset_duckdb_path(dataset)
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        try:
            con.execute("USE target;")
        except Exception:
            pass

        info = con.execute(f"PRAGMA table_info('{table}')").fetchall()
        # rows: (cid, name, type, notnull, dflt_value, pk)
        return [str(r[1]) for r in info] if info else []
    finally:
        con.close()


def _strip_trailing_sql(sql: str) -> str:
    # remove trailing semicolons/spaces; keep internal formatting
    s = sql.strip()
    while s.endswith(";"):
        s = s[:-1].rstrip()
    return s


def _normalize_sql_for_local(sql: str) -> str:
    """
    For local DuckDB execution:
    - Remove 'target.' prefix
    - Keep the rest as-is
    """
    return re.sub(r"(?i)\btarget\.", "", sql)


def _remove_qualifiers(expr_sql: str, aliases: Set[str]) -> str:
    """
    Best-effort removal of qualifiers (alias.column -> column) for locally registered DF.
    Only removes qualifiers that match known table aliases.
    """
    if not aliases:
        return expr_sql
    # \b(alias)\.(col)\b  -> col
    # Use alternation of escaped aliases
    pat = r"(?i)\b(" + "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True)) + r")\s*\.\s*([A-Za-z_][A-Za-z0-9_]*)\b"
    return re.sub(pat, r"\2", expr_sql)


def _has_aggregate_or_group(parsed: exp.Expression) -> bool:
    if parsed.find(exp.Group) is not None:
        return True
    if parsed.find(exp.Having) is not None:
        return True
    # Any aggregate function in SELECT/HAVING/etc.
    for node in parsed.walk():
        if isinstance(node, (exp.Count, exp.Sum, exp.Avg, exp.Min, exp.Max)):
            return True
    return False


def _extract_table_and_aliases(parsed: exp.Expression) -> Tuple[str, Set[str]]:
    tables = list(parsed.find_all(exp.Table))
    if not tables:
        raise ValueError("No FROM table found.")

    # single-table focus: first table
    t0 = tables[0]
    table_name = (t0.name or "").strip()
    if not table_name:
        raise ValueError("Could not determine table name from FROM.")

    # strip schema if present
    table_name = table_name.split(".")[-1]

    aliases: Set[str] = set()
    aliases.add(table_name)
    if t0.alias:
        aliases.add(t0.alias)

    return table_name, aliases


def _extract_where_clause_sql(parsed: exp.Expression) -> Optional[str]:
    w = parsed.args.get("where")
    if not w:
        return None
    # w is exp.Where; want inner expression SQL
    inner = w.this
    if not inner:
        return None
    s = inner.sql(dialect="duckdb").strip()
    return s if s else None


def _select_plain_columns_or_none(parsed: exp.Expression, table_aliases: Set[str]) -> Optional[List[str]]:
    """
    If SELECT is a list of plain columns (optionally qualified), return column names.
    If SELECT contains expressions, aggregates, stars, etc., return None.
    """
    select_exprs: Sequence[exp.Expression] = parsed.selects
    if not select_exprs:
        return None

    cols: List[str] = []
    for e in select_exprs:
        if isinstance(e, exp.Star):
            return None
        if isinstance(e, exp.Column):
            cols.append(e.name)
            continue
        # allow "DISTINCT col" encoded as exp.Distinct? sqlglot can wrap SELECT as exp.Distinct
        # but per-expression distinct often appears differently; safest: any non-Column => None
        return None

    # de-dup preserve order
    seen: Set[str] = set()
    out: List[str] = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out if out else None


def _scan_columns(parsed: exp.Expression, table_aliases: Set[str], dataset: str, table: str) -> List[str]:
    """
    Columns to fetch from LLM: all columns referenced anywhere in the query:
    SELECT columns, WHERE, GROUP BY, HAVING, ORDER BY, etc.
    If SELECT * exists, fetch full table schema.
    """
    if parsed.find(exp.Star) is not None:
        cols = _duckdb_table_columns(dataset, table)
        if cols:
            return cols
        # fallback minimal
        return ["id"]

    found: List[str] = []
    seen: Set[str] = set()

    # Collect all Column nodes
    for col in parsed.find_all(exp.Column):
        # If qualified, keep only if it matches our table/alias; if unqualified, accept (single-table assumption)
        if col.table:
            if col.table in table_aliases or col.table.lower() in {a.lower() for a in table_aliases}:
                name = col.name
            else:
                continue
        else:
            name = col.name

        if name and name not in seen:
            seen.add(name)
            found.append(name)

    if found:
        return found

    # If none found (e.g., COUNT(*) only), fetch first schema column as a payload anchor
    schema_cols = _duckdb_table_columns(dataset, table)
    if schema_cols:
        return [schema_cols[0]]
    return ["id"]


def parse_sql_query(sql: str, dataset: str) -> ParsedSQL:
    s = _strip_trailing_sql(sql)

    try:
        parsed = parse_one(s, read="duckdb")
    except Exception as e:
        raise ValueError(f"SQL parse failed (sqlglot): {e}")

    table, aliases = _extract_table_and_aliases(parsed)
    where_clause = _extract_where_clause_sql(parsed)

    has_aggr = _has_aggregate_or_group(parsed)
    select_plain = _select_plain_columns_or_none(parsed, aliases)
    scan_cols = _scan_columns(parsed, aliases, dataset, table)

    return ParsedSQL(
        sql=s,
        table=table,
        table_aliases=aliases,
        where_clause=where_clause,
        has_aggregate_or_grouping=has_aggr or (select_plain is None),
        select_plain_columns=select_plain,
        scan_columns=scan_cols,
    )


# ----------------------------
# Local post-filtering and local final query execution
# ----------------------------
def _df_from_rows(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows or [])
    # Make sure missing columns exist consistently
    return df


def apply_where_duckdb(
    rows: List[Dict[str, Any]],
    table: str,
    where_clause_sql: str,
    table_aliases: Set[str],
) -> List[Dict[str, Any]]:
    if not rows:
        return []
    if not where_clause_sql:
        return rows

    where_clean = _remove_qualifiers(where_clause_sql, table_aliases)

    con = duckdb.connect(database=":memory:")
    try:
        df = _df_from_rows(rows)
        con.register(table, df)
        q = f"SELECT * FROM {table} WHERE {where_clean}"
        out = con.execute(q).df()
        return out.to_dict(orient="records")
    finally:
        con.close()


def execute_full_sql_locally(
    rows: List[Dict[str, Any]],
    table: str,
    sql: str,
    table_aliases: Set[str],
) -> List[Dict[str, Any]]:
    """
    Execute the full SQL query locally using DuckDB against the rows retrieved from the LLM.
    This is what makes COUNT(*), GROUP BY, DISTINCT, ORDER BY, LIMIT work.
    """
    if not rows:
        # still return [] (downstream keeps alignment)
        return []

    con = duckdb.connect(database=":memory:")
    try:
        df = _df_from_rows(rows)
        con.register(table, df)

        q = _normalize_sql_for_local(sql)
        # DuckDB will not understand aliases if we register only table name; keep aliases in SQL if possible,
        # but we don't create alias bindings. Best-effort: strip alias qualifiers in SELECT/WHERE/GROUP/HAVING/ORDER
        # by replacing "<alias>." -> "" for known aliases.
        q = _remove_qualifiers(q, table_aliases)

        out = con.execute(q).df()
        return out.to_dict(orient="records")
    finally:
        con.close()


def project_rows(rows: List[Dict[str, Any]], columns: List[str]) -> List[Dict[str, Any]]:
    if not rows:
        return []
    if not columns:
        return rows
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append({c: r.get(c) for c in columns if c in r})
    return out


# ----------------------------
# Output IO
# ----------------------------
def save_galois_result_json(
    *,
    variant: str,
    dataset: str,
    query_idx: int,
    data: Dict[str, Any],
) -> None:
    """
    Saves under project root:
      galois_wo/<DATASET>/query1.json
      galois_a/<DATASET>/query1.json
      galois_s/<DATASET>/query1.json
      galois_f/<DATASET>/query1.json
    """
    variant_folder = "results_" + variant.lower()
    out_dir = Path(variant_folder) / dataset.upper()
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"query{query_idx}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    logger.info(f"[Galois] Saved: {out_path}")


# ----------------------------
# Runner
# ----------------------------
def build_variants(config: AppConfig):
    wrapper = get_llm_wrapper(config)
    provider = wrapper.get_provider_name()
    llm = wrapper.get_llm_instance()

    max_iters = getattr(getattr(config, "galois", object()), "max_iterations", 5)
    conf_thr = getattr(getattr(config, "galois", object()), "confidence_threshold", 0.6)

    variants: Dict[str, Any] = {
        "galois_wo": GaloisWOOperator(llm, max_iters=max_iters),
        "galois_a": GaloisA(llm, max_iters=max_iters),
        "galois_s": GaloisS(llm, max_iters=max_iters),
        "galois_f": GaloisF(llm, max_iters=max_iters, confidence_threshold=conf_thr),
    }
    return variants, provider


def run_galois_for_dataset(
    *,
    config: AppConfig,
    dataset: str,
    variants_to_run: List[str],
) -> None:
    ds_dir = Path("data") / dataset
    if not ds_dir.exists():
        logger.error(f"[Galois] Dataset directory does not exist: {ds_dir}")
        return

    queries = read_queries(ds_dir)
    if not queries:
        logger.warning(f"[Galois] No queries_*.sql found for dataset {dataset}")
        return

    variants, provider = build_variants(config)

    selected: Dict[str, Any] = {}
    for v in variants_to_run:
        key = v.lower()
        if key not in variants:
            logger.warning(f"[Galois] Unknown variant '{v}'. Expected one of: galois_wo/galois_a/galois_s/galois_f")
            continue
        selected[key] = variants[key]

    if not selected:
        logger.error("[Galois] No valid variants selected to run.")
        return

    logger.info(
        f"[Galois] Dataset={dataset} | provider={provider} | variants={', '.join(selected.keys())} | queries={len(queries)}"
    )

    for idx, sql in enumerate(queries, start=1):
        logger.info(f"[Galois] {dataset} Q{idx}/{len(queries)}")

        try:
            parsed = parse_sql_query(sql, dataset=dataset)
        except Exception as e:
            logger.error(f"[Galois] {dataset} Q{idx} – SQL parse failed: {e}")
            error_payload = {
                "_error": f"SQL parse failed: {e}",
                "sql": sql,
                "result_set": [],
                "time": 0.0,
                "tokens": 0,
            }
            for variant_name in selected.keys():
                save_galois_result_json(
                    variant=variant_name,
                    dataset=dataset,
                    query_idx=idx,
                    data=error_payload,
                )
            continue

        for variant_name, variant_obj in selected.items():
            t0 = time.time()
            try:
                # IMPORTANT:
                # - We ask the LLM for scan_columns (covers WHERE/GROUP/etc.)
                # - Then we compute final SQL result locally (supports COUNT(*), GROUP BY, DISTINCT, etc.)
                if isinstance(variant_obj, GaloisWOOperator):
                    res = variant_obj.scan(dataset, parsed.table, parsed.scan_columns, parsed.where_clause)
                else:
                    res = variant_obj.execute(dataset, parsed.table, parsed.scan_columns, parsed.where_clause)

                # Stable contract
                res.setdefault("result_set", [])
                res["time"] = float(res.get("time", 0.0) or 0.0)
                res["tokens"] = int(res.get("tokens", 0) or 0)
                res.setdefault("meta", {})

                # Local post-filter (WHERE) — always apply, even if variant pushed conditions (safety net)
                if parsed.where_clause:
                    res["result_set"] = apply_where_duckdb(
                        res["result_set"],
                        table=parsed.table,
                        where_clause_sql=parsed.where_clause,
                        table_aliases=parsed.table_aliases,
                    )

                # Local final execution of the full SQL to handle aggregates / group / distinct / order / limit
                # If the SELECT is plain columns only and has no grouping/aggregate, this is still valid,
                # but could reorder / distinct, etc. so we do it always.
                res["result_set"] = execute_full_sql_locally(
                    res["result_set"],
                    table=parsed.table,
                    sql=parsed.sql,
                    table_aliases=parsed.table_aliases,
                )

                # If SELECT is plain columns, ensure projection matches requested columns (order-preserving)
                if parsed.select_plain_columns:
                    res["result_set"] = project_rows(res["result_set"], parsed.select_plain_columns)

                # meta
                res["meta"].setdefault("dataset", dataset)
                res["meta"].setdefault("table", parsed.table)
                res["meta"].setdefault("query_idx", idx)
                res["meta"].setdefault("provider", provider)

                payload = {
                    "sql": sql,
                    "parsed": {
                        "table": parsed.table,
                        "scan_columns": parsed.scan_columns,
                        "select_plain_columns": parsed.select_plain_columns,
                        "where_clause": parsed.where_clause,
                    },
                    **res,
                }

            except Exception as e:
                logger.error(f"[Galois] {dataset} Q{idx} – {variant_name} failed: {e}")
                payload = {
                    "_error": str(e),
                    "sql": sql,
                    "parsed": {
                        "table": parsed.table,
                        "scan_columns": parsed.scan_columns,
                        "select_plain_columns": parsed.select_plain_columns,
                        "where_clause": parsed.where_clause,
                    },
                    "result_set": [],
                    "time": 0.0,
                    "tokens": 0,
                }
            finally:
                # If upstream returned time=0 due to error, still keep runner overhead out of payload.
                _ = time.time() - t0

            save_galois_result_json(
                variant=variant_name,
                dataset=dataset,
                query_idx=idx,
                data=payload,
            )


def main() -> None:
    """
    Examples:
      python -m src.galois.run_galois
      python -m src.galois.run_galois --datasets flight-2 imdb --variants galois_wo galois_f
    """
    ap = argparse.ArgumentParser(
        description="Run Galois variants over SQL datasets and save results to galois_* folders."
    )
    ap.add_argument(
        "--datasets",
        nargs="*",
        help=(
            "Optional list of dataset folder names under ./data. "
            "If omitted, all datasets discovered in ./data are used."
        ),
    )
    ap.add_argument(
        "--variants",
        nargs="*",
        default=["galois_wo", "galois_a", "galois_s", "galois_f"],
        help="Variants to run: galois_wo galois_a galois_s galois_f",
    )

    args = ap.parse_args()
    config = load_config()

    datasets = args.datasets or discover_datasets()
    if not datasets:
        logger.error("No datasets found under ./data")
        return

    logger.info(
        f"[Galois] Running for datasets: {', '.join(datasets)} | variants={', '.join(args.variants)}"
    )

    for ds in datasets:
        run_galois_for_dataset(
            config=config,
            dataset=ds,
            variants_to_run=args.variants,
        )


if __name__ == "__main__":
    main()
