import duckdb, os, sys, glob, pathlib

def init_dataset(dataset: str):
    DATASET = dataset.upper()
    dataset_lower = dataset.lower()
    os.makedirs("db", exist_ok=True)
    db_path = f"db/{DATASET}.duckdb"

    data_dir = pathlib.Path("data") / dataset_lower
    if not data_dir.is_dir():
        raise SystemExit(f"Data folder not found: {data_dir}")

    con = duckdb.connect(db_path)
    try:
        ingest_sql = data_dir / f"ingest_{dataset_lower}.sql"
        if ingest_sql.is_file():
            # ingestion from dataset
            con.execute(open(ingest_sql, encoding="utf-8").read())
            print(f"Executed ingest script: {ingest_sql}")
        else:
            # Fallback: load CSV automatically
            for csv in glob.glob(str(data_dir / "*.csv")):
                table = pathlib.Path(csv).stem
                con.execute(f"""
                    CREATE TABLE IF NOT EXISTS "{table}" AS
                    SELECT * FROM read_csv_auto(?, header=True);
                """, [csv])
            print(f"ℹNo ingest_{dataset_lower}.sql found, loaded CSVs instead.")
        print(f"Initialized {db_path} from {data_dir}")
    finally:
        con.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/init_duckdb.py <DATASET_NAME>")
        sys.exit(1)
    init_dataset(sys.argv[1])
