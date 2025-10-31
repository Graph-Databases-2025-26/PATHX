from __future__ import annotations
import os
from pathlib import Path
import duckdb

# Directory where all .duckdb files will be stored
DB_DIR = Path("db")
DB_DIR.mkdir(exist_ok=True)

def connect_dataset_db(dataset: str, persist: bool = True) -> duckdb.DuckDBPyConnection:
    """
    Connect to the DuckDB database corresponding to a specific dataset.

    """
    if persist:
        # Each dataset gets its own file: db/MOVIES.duckdb, db/GEO.duckdb, etc.
        db_path = DB_DIR / f"{dataset.upper()}.duckdb"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(db_path))
    # In-memory database (disappears when closed)
    return duckdb.connect()


def _exec_sql_file(con: duckdb.DuckDBPyConnection, sql_path: Path, workdir: Path) -> None:
    """
    Execute an SQL file as-is. Temporarily change working directory to the dataset folder
    so relative paths inside COPY statements (like COPY 'table.csv'...) resolve correctly.

    """
    sql_text = sql_path.read_text(encoding="utf-8")
    cwd = Path.cwd()
    try:
        os.chdir(workdir)
        con.execute(sql_text)   # execute multiple statements in one call
    finally:
        os.chdir(cwd)


def init_schema(con: duckdb.DuckDBPyConnection, schema: str = "target", clean: bool = True) -> None:
    """
    Create (and optionally reset) the schema for a dataset.

    """
    if clean:
        # Drop the schema completely to ensure a clean environment
        con.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE;')
    con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";')


def _table_exists(con, schema: str, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_schema=? AND table_name=?",
        [schema, name]
    ).fetchone()
    return row is not None

def ingest_dataset(con: duckdb.DuckDBPyConnection, dataset_dir: Path, schema: str = "target") -> None:
    init_schema(con, schema=schema, clean=True)

    ingest_files = sorted(dataset_dir.glob("ingest*.sql")) or sorted(dataset_dir.glob("ingest_*.sql"))
    if ingest_files:
        _exec_sql_file(con, ingest_files[0], workdir=dataset_dir)
    else:
        for csv in sorted(dataset_dir.glob("*.csv")):
            table = csv.stem
            con.execute(
                f'CREATE OR REPLACE TABLE "{schema}"."{table}" AS '
                'SELECT * FROM read_csv_auto(?, header=True)',
                [str(csv)]
            )

    # ---- Dataset-specific compatibility aliases (SAFE) ----
    if dataset_dir.name == "fortune":
        # Case A: CSV fallback created target.fortune1000_2024, queries expect fortune_2024
        if _table_exists(con, schema, "fortune1000_2024") and not _table_exists(con, schema, "fortune_2024"):
            con.execute(f'CREATE OR REPLACE VIEW "{schema}".fortune_2024 AS SELECT * FROM "{schema}".fortune1000_2024')
        # Case B: ingest already created target.fortune_2024 -> nothing to do
        # (If someone did the opposite mapping, you can add the reverse alias here if ever needed.)
