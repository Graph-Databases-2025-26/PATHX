from __future__ import annotations

import os
from pathlib import Path
import duckdb

DB_DIR = Path("db")
DB_DIR.mkdir(exist_ok=True)


class DuckDB:
    """
    Lightweight wrapper around DuckDB for schema inspection
    (used to build {{ schema_info }} for prompts).
    """

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: duckdb.DuckDBPyConnection | None = None

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        if self._conn is None:
            self._conn = duckdb.connect(str(self.db_path))
        return self._conn

    def list_tables(self, schema: str = "target") -> list[str]:
        con = self.get_connection()
        rows = con.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema=?
            ORDER BY table_name
            """,
            [schema],
        ).fetchall()
        return [r[0] for r in rows]

    def describe_table(self, table: str, schema: str = "target") -> list[tuple]:
        """
        Returns list of (column name, column type)
        """
        con = self.get_connection()
        return con.execute(
            """
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_schema=? AND table_name=?
            ORDER BY ordinal_position
            """,
            [schema, table],
        ).fetchall()

    def get_table_info(self, schema: str = "target") -> str:
        """
        Returns a formatted multiline string describing all tables.
        This will be injected into jinja2 prompts as {{ schema_info }}.
        """
        tables = self.list_tables(schema)
        if not tables:
            return "(no tables found)"

        out_lines: list[str] = []
        for t in tables:
            out_lines.append(f"TABLE {t}:")
            cols = self.describe_table(t, schema)
            for col_name, col_type in cols:
                out_lines.append(f"  - {col_name}: {col_type}")
            out_lines.append("")  # blank line

        return "\n".join(out_lines).strip()

    def path(self) -> Path:
        return self.db_path


def connect_dataset_db(dataset: str, persist: bool = True) -> duckdb.DuckDBPyConnection:
    """
    If persist=True then use db/FLIGHT-2.duckdb
    Else then in-memory DB.
    """
    if persist:
        db_path = DB_DIR / f"{dataset.upper()}.duckdb"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return duckdb.connect(str(db_path))
    return duckdb.connect()


def _exec_sql_file(con: duckdb.DuckDBPyConnection, sql_path: Path, workdir: Path) -> None:
    sql_text = sql_path.read_text(encoding="utf-8")
    cwd = Path.cwd()
    try:
        os.chdir(workdir)
        con.execute(sql_text)
    finally:
        os.chdir(cwd)


def init_schema(con: duckdb.DuckDBPyConnection, schema: str = "target", clean: bool = True) -> None:
    if clean:
        con.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE;')
    con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}";')


def _table_exists(con, schema: str, name: str) -> bool:
    row = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_schema=? AND table_name=?",
        [schema, name],
    ).fetchone()
    return row is not None


def ingest_dataset(con: duckdb.DuckDBPyConnection, dataset_dir: Path, schema: str = "target") -> None:
    """
    Loads CSV or ingest_*.sql into DuckDB.
    """
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
                [str(csv)],
            )

    # Special handling for "fortune"
    if dataset_dir.name == "fortune":
        if _table_exists(con, schema, "fortune1000_2024") and not _table_exists(con, schema, "fortune_2024"):
            con.execute(
                f'CREATE OR REPLACE VIEW "{schema}".fortune_2024 AS '
                f'SELECT * FROM "{schema}".fortune1000_2024'
            )


def get_duckdb_for_schema(dataset: str) -> DuckDB:
    """
    Returns a DuckDB wrapper for dataset DB, located at db/<DATASET>.duckdb.
    Used by baselines to build {{ schema_info }}.
    """
    db_path = DB_DIR / f"{dataset.upper()}.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    if not db_path.exists():
        duckdb.connect(str(db_path)).close()

    return DuckDB(db_path)
