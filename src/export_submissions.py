import os, sys, json, time, re, glob, pathlib
import duckdb

def connect_db(dataset: str):
    return duckdb.connect(f"db/{dataset.upper()}.duckdb")

def rows_to_objects(cur, rows):
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in rows]

def split_queries_from_monofile(text: str):

    # First try by markdowns --queryN
    blocks = re.split(r"(?im)^\s*--\s*query\s*\d+\s*$", text)
    parts = [b.strip() for b in blocks if b.strip()]
    if len(parts) > 0 and len(blocks) > 1:
        return [p if p.rstrip().endswith(';') else p + ';' for p in parts]

    # Folkback: just by ';'
    parts = [q.strip() for q in text.split(';') if q.strip()]
    return [p + ';' for p in parts]

def find_queries(dataset: str):
    DATASET = dataset.upper()
    dataset_lower = dataset.lower()

    # Option A: queries/<DATASET>/query*.sql
    qdir = pathlib.Path("queries") / DATASET
    files = sorted(glob.glob(str(qdir / "query*.sql")))
    if files:
        return ("files", files)

    # Option B: data/<dataset>/queries_<dataset>.sql
    monofile = pathlib.Path("data") / dataset_lower / f"queries_{dataset_lower}.sql"
    if monofile.is_file():
        return ("mono", monofile)

    raise SystemExit(f"No queries found in queries/{DATASET} or {monofile}")

def export_dataset(dataset: str, fmt: str = "simple"):
    DATASET = dataset.upper()
    outdir = pathlib.Path("submissions") / DATASET
    outdir.mkdir(parents=True, exist_ok=True)

    kind, src = find_queries(dataset)
    con = connect_db(dataset)

    try:
        if kind == "files":
            sources = [(i+1, pathlib.Path(p).read_text(encoding="utf-8"))
                       for i, p in enumerate(sorted(src))]
        else:
            text = pathlib.Path(src).read_text(encoding="utf-8")
            parts = split_queries_from_monofile(text)
            sources = [(i+1, q) for i, q in enumerate(parts)]

        for idx, sql in sources:
            t0 = time.perf_counter()
            cur = con.execute(sql)
            rows = cur.fetchall()
            result_set = rows_to_objects(cur, rows)
            dt = time.perf_counter() - t0

            payload = result_set if fmt == "simple" else {
                "result_set": result_set,
                "time": round(dt, 6)
                # "tokens": <add later if needed>
            }

            out_json = outdir / f"query{idx}.json"
            with open(out_json, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f"{DATASET}/query{idx}.json written ({fmt})")
    finally:
        con.close()

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python -m src.export_submissions <DATASET_NAME> [simple|full]")
        sys.exit(1)
    dataset = sys.argv[1]
    fmt = sys.argv[2] if len(sys.argv) == 3 else "simple"
    export_dataset(dataset, fmt)
