from __future__ import annotations
import argparse, json, time
from pathlib import Path
import duckdb

from ..db.duckdb_client import connect_dataset_db, ingest_dataset

DATA = Path("data")
SUB  = Path("submissions")

def read_queries(ds_dir: Path) -> list[str]:
    import re
    qfile = sorted(ds_dir.glob("queries*.sql")) or sorted(ds_dir.glob("queries_*.sql"))
    if not qfile:
        raise FileNotFoundError(f"No queries_*.sql found in {ds_dir}")
    text = qfile[0].read_text(encoding="utf-8")
    lines = [l for l in text.splitlines() if not l.strip().startswith("--")]
    text = "\n".join(lines)
    parts = [p.strip() for p in re.split(r';\s*(?:--.*)?\n', text) if p.strip()]
    stmts, seen = [], set()
    for p in parts:
        for s in [x.strip() for x in p.split(";") if x.strip()]:
            if s and s not in seen:
                seen.add(s)
                stmts.append(s)
    return stmts

def run_queries(dataset: str, schema="target"):
    ds_dir = DATA / dataset
    out_dir = SUB / dataset.upper()
    out_dir.mkdir(parents=True, exist_ok=True)

    con = connect_dataset_db(dataset, persist=True)
    ingest_dataset(con, ds_dir, schema=schema)

    for i, sql in enumerate(read_queries(ds_dir), 1):
        t0 = time.time()
        try:
            df = con.execute(sql).df()
            rows = df.to_dict(orient="records")
            err = None
        except Exception as e:
            rows, err = [], str(e)
        elapsed = round(time.time() - t0, 6)
        payload = {"result_set": rows, "time": elapsed, "tokens": 0}
        (out_dir / f"query{i}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"[{dataset}] query{i}: {len(rows)} rows, {elapsed}s {'OK' if not err else err}")

    con.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    args = ap.parse_args()
    for ds in args.datasets:
        run_queries(ds)

if __name__ == "__main__":
    main()
