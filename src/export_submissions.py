import os, sys, json, time, glob

from src.db_utils import connect_db, run_query

def export_dataset(dataset: str, fmt: str = "simple"):
    dataset = dataset.upper()
    qdir = f"queries/{dataset}"
    outdir = f"submissions/{dataset}"
    os.makedirs(outdir, exist_ok=True)

    # read *.sql by name: query1.sql, query2.sql, ...
    files = sorted(glob.glob(os.path.join(qdir, "query*.sql")))
    if not files:
        raise SystemExit(f"No queries found in {qdir}")

    con = connect_db(dataset)
    try:
        for path in files:
            qname = os.path.splitext(os.path.basename(path))[0]   # queryN
            with open(path, "r", encoding="utf-8") as f:
                sql = f.read().strip()
            t0 = time.perf_counter()
            result_set = run_query(con, sql)
            dt = time.perf_counter() - t0

            if fmt == "full":
                payload = {
                    "result_set": result_set,
                    "time": round(dt, 6),
                    # tokens can be readed/added later; optionally:
                    # "tokens": <int>
                }
            else:
                payload = result_set  # simple format

            out_json = os.path.join(outdir, f"{qname}.json")
            with open(out_json, "w", encoding="utf-8") as out:
                json.dump(payload, out, ensure_ascii=False, indent=2)

            print(f"{dataset}/{qname}.json written ({fmt})")
    finally:
        con.close()

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python -m src.export_submissions <DATASET_NAME> [simple|full]")
        sys.exit(1)
    dataset = sys.argv[1]
    fmt = sys.argv[2] if len(sys.argv) == 3 else "simple"
    export_dataset(dataset, fmt)
