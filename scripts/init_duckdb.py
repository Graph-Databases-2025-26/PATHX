import duckdb, os, sys, glob

def init_dataset(dataset: str):
    dataset = dataset.upper()
    os.makedirs("db", exist_ok=True)
    db_path = f"db/{dataset}.duckdb"
    data_dir = f"data/{dataset}"

    if not os.path.isdir(data_dir):
        raise SystemExit(f"Data folder not found: {data_dir}")

    con = duckdb.connect(db_path)
    try:
        # Загружаем все CSV в отдельные таблицы по имени файла
        for csv in glob.glob(os.path.join(data_dir, "*.csv")):
            table = os.path.splitext(os.path.basename(csv))[0]
            con.execute(f"""
                CREATE TABLE IF NOT EXISTS "{table}" AS
                SELECT * FROM read_csv_auto(?, header=True);
            """, [csv])
        print(f"Initialized {db_path} from {data_dir}")
    finally:
        con.close()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/init_duckdb.py <DATASET_NAME>")
        sys.exit(1)
    init_dataset(sys.argv[1])
