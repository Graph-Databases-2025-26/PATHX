import argparse, duckdb
from pathlib import Path
from utils.io_utils import write_json, read_sql_statements
from utils.timers import timer

DATA = Path("data")
SUB = Path("submissions")

def load_dataset_to_duckdb(con: duckdb.DuckDBPyConnection, ds_dir: Path, schema: str = "target"):
    con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    for csv in sorted(ds_dir.glob("*.csv")):
        tbl = csv.stem
        con.execute(
            f'CREATE OR REPLACE TABLE "{schema}"."{tbl}" AS '
            'SELECT * FROM read_csv_auto(?, header=True)',
            [str(csv)]
        )
    if ds_dir.name == "fortune":
        con.execute(f'CREATE OR REPLACE VIEW "{schema}".fortune_2024 AS SELECT * FROM "{schema}".fortune1000_2024')

def run_dataset(ds: str, schema: str = "target"):
    ds_dir = DATA / ds
    qfiles = sorted(ds_dir.glob("queries_*.sql"))
    if not qfiles:
        print(f"[WARN] no queries_*.sql in {ds_dir}")
        return
    queries = read_sql_statements(qfiles[0])

    out_dir = SUB / ds.upper()
    out_dir.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()
    load_dataset_to_duckdb(con, ds_dir, schema)

    for i, sql in enumerate(queries, 1):
        with timer() as elapsed:
            df = con.execute(sql).df()
        payload = {
            "result_set": df.to_dict(orient="records"),
            "time": round(elapsed(), 6),
            "tokens": 0   # pure SQL baseline
        }
        write_json(out_dir / f"query{i}.json", payload)
        print(f"[SQL] {ds} query{i}: {len(payload['result_set'])} rows, {payload['time']} s")

    con.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", required=True,
                    help="e.g., movies world geo presidents fortune premier flight-2 flight-4")
    args = ap.parse_args()
    for ds in args.datasets:
        run_dataset(ds)

if __name__ == "__main__":
    main()
