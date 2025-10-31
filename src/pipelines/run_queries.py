from __future__ import annotations
import argparse, json, time
from pathlib import Path

from ..db.duckdb_client import connect_dataset_db, ingest_dataset
from .common import discover_datasets, read_queries

DATA = Path("data")
SUB  = Path("submissions")

def run_queries(dataset: str, schema="target"):
    ds_dir = DATA / dataset
    out_dir = SUB / dataset.upper()
    out_dir.mkdir(parents=True, exist_ok=True)

    con = connect_dataset_db(dataset, persist=True)
    ingest_dataset(con, ds_dir, schema=schema)

    for i, sql in enumerate(read_queries(ds_dir), 1):
        t0 = time.time()
        rows, err = [], None
        try:
            df = con.execute(sql).df()
            rows = df.to_dict(orient="records")
        except Exception as e:
            err = str(e)
        elapsed = round(time.time() - t0, 6)
        payload = {"result_set": rows, "time": elapsed, "tokens": 0}
        (out_dir / f"query{i}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"[{dataset}] query{i}: {len(rows)} rows, {elapsed}s {'OK' if not err else err}")

    con.close()

def main():
    ap = argparse.ArgumentParser(description="Execute queries_*.sql for datasets (auto if not provided)")
    ap.add_argument("--datasets", nargs="*", help="Optional list; if omitted, process all in ./data")
    args = ap.parse_args()

    datasets = args.datasets or discover_datasets()
    print(f"[INFO] Datasets: {', '.join(datasets)}")
    for ds in datasets:
        run_queries(ds)

if __name__ == "__main__":
    main()
