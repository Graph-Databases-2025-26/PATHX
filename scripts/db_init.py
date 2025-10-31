#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Database Initialization and CSV Ingestion Module

This module handles:
- Creating DuckDB database from CSV files
- Loading CSV data into DuckDB tables
- Creating schemas and compatibility aliases
- Persisting the database to .duckdb file

Usage:
    python db_init.py --data-root ./data --db-path ./database.duckdb --schema-name target
"""

import argparse
import sys
from pathlib import Path
import duckdb


def find_dataset_dirs(data_root: Path):
    """
    Find all dataset directories in the data root folder.
    
    Args:
        data_root: Path to the root data directory
        
    Returns:
        Dictionary mapping dataset names to their directory paths
    """
    # Accept both data_root/ and data_root/data/
    root = data_root / "data" if (data_root / "data").exists() else data_root
    out = {}
    
    for ds in sorted(root.iterdir()):
        if not ds.is_dir():
            continue
        if ds.name == "flight":
            for sub in sorted(ds.iterdir()):
                if sub.is_dir():
                    out[f"flight-{sub.name.split('-')[-1]}"] = sub
        else:
            out[ds.name] = ds
    
    return out


def load_csvs_into_duckdb(con: duckdb.DuckDBPyConnection, ds_dir: Path, schema: str):
    """
    Load all CSV files from a dataset directory into DuckDB tables.
    
    Args:
        con: DuckDB connection object
        ds_dir: Path to dataset directory containing CSV files
        schema: Schema name to create tables in
    """
    con.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
    
    for csv_path in sorted(ds_dir.glob("*.csv")):
        table = csv_path.stem
        
        if table == "movies":
            # Keep schema variable consistent everywhere
            con.execute(f'''
            CREATE SCHEMA IF NOT EXISTS "{schema}";
            CREATE OR REPLACE TABLE "{schema}".movies(
                primarytitle VARCHAR,
                originaltitle VARCHAR,
                startyear BIGINT,
                endyear BIGINT,
                runtimeminutes BIGINT,
                genres VARCHAR,
                director VARCHAR,
                birthyear BIGINT,
                deathyear BIGINT
            );
            ''')
            
            # Use a parameter for the path and let DuckDB handle quoting
            con.execute(
                f'''COPY "{schema}".movies FROM ?
                (FORMAT CSV, HEADER TRUE, DELIMITER ',', QUOTE '"', ESCAPE '"', NULL 'null');''',
                [str(csv_path)]
            )
        else:
            con.execute(
                f'CREATE OR REPLACE TABLE "{schema}"."{table}" AS '
                'SELECT * FROM read_csv_auto(?, header=True)',
                [str(csv_path)]
            )


def add_compat_aliases(con: duckdb.DuckDBPyConnection, schema: str, dataset_name: str):
    """
    Add compatibility aliases for tables with different naming conventions.
    
    Args:
        con: DuckDB connection object
        schema: Schema name containing the tables
        dataset_name: Name of the dataset to check for aliases
    """
    # Aliases needed because some SQLs reference slightly different table names than CSV file names.
    aliases = []
    
    if dataset_name == "fortune":
        # CSV is fortune1000_2024, SQL uses fortune_2024
        aliases.append(("fortune_2024", "fortune1000_2024"))
    # You can extend here if more mismatches appear.
    
    for alias, real in aliases:
        # Create view alias only if real table exists
        exists = con.execute(
            "SELECT 1 FROM information_schema.tables WHERE table_schema=? AND table_name=?",
            [schema, real]
        ).fetchone()
        
        if exists:
            con.execute(f'CREATE OR REPLACE VIEW "{schema}"."{alias}" AS SELECT * FROM "{schema}"."{real}"')


def initialize_database(data_root: Path, db_path: Path, schema_name: str = "target", datasets=None):
    """
    Initialize DuckDB database and load CSV data.
    
    Args:
        data_root: Path to root directory containing datasets
        db_path: Path where the .duckdb file will be created
        schema_name: Name of the schema to create (default: "target")
        datasets: Optional list of specific datasets to load
        
    Returns:
        Dictionary with statistics about loaded datasets
    """
    ds_dirs = find_dataset_dirs(data_root)
    
    if datasets:
        ds_dirs = {k: v for k, v in ds_dirs.items() if k in datasets}
    
    # Create or connect to the database file
    con = duckdb.connect(str(db_path))
    
    stats = {}
    
    for ds_name, ds_dir in ds_dirs.items():
        print(f"[INFO] Loading dataset: {ds_name}")
        
        # Load CSVs into the schema
        load_csvs_into_duckdb(con, ds_dir, schema_name)
        
        # Add dataset-specific aliases
        add_compat_aliases(con, schema_name, ds_name)
        
        # Count tables loaded
        table_count = con.execute(
            "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema=?",
            [schema_name]
        ).fetchone()[0]
        
        stats[ds_name] = {"tables_loaded": table_count}
        print(f"[OK] {ds_name}: loaded into schema '{schema_name}'")
    
    con.close()
    
    print(f"\n[SUCCESS] Database initialized at: {db_path}")
    return stats


def main():
    """Main entry point for database initialization."""
    ap = argparse.ArgumentParser(description="Initialize DuckDB database from CSV files")
    ap.add_argument("--data-root", type=Path, required=True, help="Path to data directory")
    ap.add_argument("--db-path", type=Path, required=True, help="Path for output .duckdb file")
    ap.add_argument("--datasets", nargs="*", default=None, help="Specific datasets to load (optional)")
    ap.add_argument("--schema-name", default="target", help='DuckDB schema name (default: "target")')
    
    args = ap.parse_args()
    
    try:
        stats = initialize_database(
            data_root=args.data_root,
            db_path=args.db_path,
            schema_name=args.schema_name,
            datasets=args.datasets
        )
        
        print("\n=== Database Statistics ===")
        for ds_name, ds_stats in stats.items():
            print(f"{ds_name}: {ds_stats['tables_loaded']} tables")
            
    except duckdb.IOException as e:
        print(f"[ERROR] DuckDB I/O error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
