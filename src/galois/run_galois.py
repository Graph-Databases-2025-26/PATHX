# src/galois/run_galois.py
from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
# SQL parsing (minimal, robust-enough for project queries)
# ----------------------------
@dataclass(frozen=True)
class ParsedSQL:
    table: str
    attributes: List[str]
    where_clause: Optional[str]


_SQL_SELECT_RE = re.compile(
    r"(?is)^\s*select\s+(?P<select>.*?)\s+from\s+(?P<from>.*?)(?:\s+where\s+(?P<where>.*?))?\s*;?\s*$"
)


def _split_select_list(select_part: str) -> List[str]:
    """
    Split SELECT list by commas, respecting parentheses and quotes.
    Keeps simple identifiers; drops aliases.
    """
    s = select_part.strip()
    if s == "*":
        raise ValueError("SELECT * is not supported for Galois runner; list attributes explicitly.")

    items: List[str] = []
    cur: List[str] = []
    depth = 0
    in_sq = False
    in_dq = False

    for ch in s:
        if ch == "'" and not in_dq:
            in_sq = not in_sq
        elif ch == '"' and not in_sq:
            in_dq = not in_dq
        elif not in_sq and not in_dq:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth = max(0, depth - 1)

        if ch == "," and depth == 0 and not in_sq and not in_dq:
            items.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)

    if cur:
        items.append("".join(cur).strip())

    # Normalize: take last identifier after dot, drop aliases (best-effort)
    attrs: List[str] = []
    for it in items:
        # remove `AS alias` or trailing alias
        it2 = re.sub(r"(?is)\s+as\s+\w+\s*$", "", it).strip()
        it2 = re.sub(r"(?is)\s+\w+\s*$", lambda m: "" if re.search(r"[()*/+-]", it2) else m.group(0), it2).strip()

        # If expressions exist, we cannot map to a single attribute reliably
        if re.search(r"[()*/+-]", it2):
            raise ValueError(f"Unsupported SELECT expression: '{it}'. Use plain column names.")

        it2 = it2.strip("`")
        # table.col -> col
        it2 = it2.split(".")[-1].strip()
        if not it2:
            continue
        attrs.append(it2)

    if not attrs:
        raise ValueError(f"Could not parse attributes from SELECT list: '{select_part}'")

    return attrs


def parse_sql_query(sql: str) -> ParsedSQL:
    """
    Parse a single SQL query of the form:
      SELECT a, b FROM table WHERE cond...
    """
    m = _SQL_SELECT_RE.match(sql.strip())
    if not m:
        raise ValueError(f"Unsupported SQL format (expected SELECT ... FROM ... [WHERE ...]): {sql[:120]}...")

    select_part = m.group("select")
    from_part = m.group("from").strip()
    where_part = m.group("where")
    where_clause = where_part.strip() if where_part and where_part.strip() else None

    # Extract table name (first token, strip schema if present)
    # Handles: FROM city; FROM schema.city; FROM "city"
    table_token = from_part.split()[0].strip().strip(";")
    table_token = table_token.strip('"').strip("`")
    table = table_token.split(".")[-1].strip()

    attrs = _split_select_list(select_part)
    return ParsedSQL(table=table, attributes=attrs, where_clause=where_clause)


# ----------------------------
# Output IO
# ----------------------------
def save_galois_result_json(
    *,
    variant: str,
    dataset: str,
    provider: str,
    query_idx: int,
    data: Dict[str, Any],
) -> None:
    """
    Saves under project root:
      results_galois_wo/<DATASET>/<provider>/query1.json
      results_galois_a/<DATASET>/<provider>/query1.json
      ...
    """
    variant_folder = variant.lower()
    out_dir = Path(variant_folder) / dataset
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

    # Keep iteration semantics consistent across all scans
    max_iters = getattr(config.galois, "max_iterations", 5)

    variants = {
        "results_galois_wo": GaloisWOOperator(llm, max_iters=max_iters),
        "results_galois_a": GaloisA(llm, max_iters=max_iters),
        "results_galois_s": GaloisS(llm, max_iters=max_iters),
        "results_galois_f": GaloisF(
            llm,
            max_iters=max_iters,
            confidence_threshold=getattr(config.galois, "confidence_threshold", 0.6),
        ),
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

    # filter variants
    selected: Dict[str, Any] = {}
    for v in variants_to_run:
        key = f"results_{v.lower()}" if not v.lower().startswith("results_") else v.lower()
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
            parsed = parse_sql_query(sql)
        except Exception as e:
            logger.error(f"[Galois] {dataset} Q{idx} – SQL parse failed: {e}")
            error_payload = {
                "_error": f"SQL parse failed: {e}",
                "sql": sql,
                "result_set": [],
                "time": 0.0,
                "tokens": 0,
            }
            # Save parse failure for all selected variants (so downstream tables remain aligned)
            for variant_name in selected.keys():
                save_galois_result_json(
                    variant=variant_name,
                    dataset=dataset,
                    provider=provider,
                    query_idx=idx,
                    data=error_payload,
                )
            continue

        for variant_name, variant_obj in selected.items():
            try:
                if isinstance(variant_obj, GaloisWOOperator):
                    res = variant_obj.scan(dataset, parsed.table, parsed.attributes, parsed.where_clause)
                else:
                    # A/S/F variants expose execute(...)
                    res = variant_obj.execute(dataset, parsed.table, parsed.attributes, parsed.where_clause)

                # Enforce stable contract (avoid None in tables)
                res.setdefault("result_set", [])
                res["time"] = float(res.get("time", 0.0) or 0.0)
                res["tokens"] = int(res.get("tokens", 0) or 0)
                res.setdefault("meta", {})
                res["meta"].setdefault("dataset", dataset)
                res["meta"].setdefault("table", parsed.table)
                res["meta"].setdefault("query_idx", idx)

                payload = {
                    "sql": sql,
                    "parsed": {
                        "table": parsed.table,
                        "attributes": parsed.attributes,
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
                        "attributes": parsed.attributes,
                        "where_clause": parsed.where_clause,
                    },
                    "result_set": [],
                    "time": 0.0,
                    "tokens": 0,
                }

            save_galois_result_json(
                variant=variant_name,
                dataset=dataset,
                provider=provider,
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
        description="Run Galois variants over SQL datasets and save results to results_galois_* folders."
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
