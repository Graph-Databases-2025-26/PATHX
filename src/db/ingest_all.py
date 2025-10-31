
from __future__ import annotations
import argparse
from pathlib import Path
import duckdb

from duckdb_client import connect_dataset_db, ingest_dataset

DATA = Path("data")

def discover_datasets() -> list[str]:
    """
    Auto-discover dataset folders in ./data (one level).
    Expected: data/<dataset>/ containing CSV + ingest_*.sql + queries_*.sql
    """
    out: list[str] = []
    for p in sorted(DATA.iterdir()):
        if p.is_dir():
            out.append(p.name)
    return out

def ingest_one(dataset: str, schema: str = "target", persist_db: bool = True) -> None:
    """
    Create/overwrite a DuckDB database for the dataset and load all tables via ingest_*.sql (or CSV fallback).
    """
    ds_dir = DATA / dataset
    if not ds_dir.exists():
        print(f"[WARN] dataset folder not found: {ds_dir}")
        return

    con = connect_dataset_db(dataset, persist=persist_db)
    try:
        ingest_dataset(con, ds_dir, schema=schema)  # drops+creates schema, runs ingest*.sql or auto-CSV
        # Optional: quick verification (list tables)
        tbls = con.execute("SHOW TABLES FROM {}".format(schema)).fetchall()
        names = ", ".join(t[0] for t in tbls)
        print(f"[OK] {dataset}: initialized schema '{schema}' with tables: {names or '(none)'}")
    finally:
        con.close()

def main():
    ap = argparse.ArgumentParser(description="Initialize per-dataset DuckDB databases from CSV/ingest_*.sql")
    ap.add_argument("--datasets", nargs="*", help="Datasets to ingest; if omitted, auto-discover all under ./data")
    ap.add_argument("--schema", default="target", help="Schema name to (re)create for each dataset (default: target)")
    ap.add_argument("--no-persist", action="store_true", help="Use in-memory DB instead of file-backed .duckdb")
    args = ap.parse_args()

    datasets = args.datasets or discover_datasets()
    print(f"[INFO] Datasets to ingest: {', '.join(datasets)}")

    for ds in datasets:
        ingest_one(ds, schema=args.schema, persist_db=not args.no_persist)

if __name__ == "__main__":
    main()
