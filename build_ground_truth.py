#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build Ground Truth - Main Pipeline Script

This script combines the three modular components (db_init, db_query, db_explain)
to replicate the original build_ground_truth.py functionality while maintaining
the separation of concerns.

Usage (compatible with original):
    python build_ground_truth.py --data-root data --ground-root ground_truth_result \
        --datasets flight-2 flight-4 fortune geo premier presidents world movies
"""

import argparse
import json
import sys
from pathlib import Path
import duckdb

# Import the modular components
from db_init import find_dataset_dirs, load_csvs_into_duckdb, add_compat_aliases
from db_query import read_sql_statements


def build_ground_truth_for_dataset(ds_name: str, ds_dir: Path, output_root: Path, schema_name: str = "target"):
    """
    Build ground truth for a single dataset.
    
    Args:
        ds_name: Name of the dataset
        ds_dir: Path to dataset directory containing CSV and SQL files
        output_root: Root directory for output files
        schema_name: Schema name to use in DuckDB
    """
    # Find query files
    qfiles = list(ds_dir.glob("queries_*.sql"))
    if not qfiles:
        print(f"[WARN] No queries_*.sql in {ds_dir}; skipping")
        return
    
    qpath = sorted(qfiles)[0]
    sqls = read_sql_statements(qpath)
    
    if not sqls:
        print(f"[WARN] No SQL statements in {qpath}")
        return
    
    # Create output directory (uppercase dataset name)
    out_dir = output_root / ds_name.upper()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Create in-memory DuckDB connection
    con = duckdb.connect()
    
    try:
        # Load CSVs into the schema
        load_csvs_into_duckdb(con, ds_dir, schema_name)
        
        # Add dataset-specific aliases
        add_compat_aliases(con, schema_name, ds_name)
        
        # Execute queries and save results
        for i, sql in enumerate(sqls, start=1):
            try:
                df = con.execute(sql).df()
                rows = df.to_dict(orient="records")
            except Exception as e:
                print(f"[ERROR] {ds_name} q{i}: {e}\n")
                rows = []
            
            # Write JSON output
            output_file = out_dir / f"query{i}.json"
            output_file.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(f"[OK] {ds_name} query{i}: {len(rows)} rows")
            
    finally:
        con.close()


def main():
    """Main entry point - replicates original build_ground_truth.py behavior."""
    ap = argparse.ArgumentParser(
        description="Build ground-truth by executing queries over CSVs using DuckDB"
    )
    ap.add_argument("--data-root", type=Path, required=True,
                    help="Root directory containing dataset folders")
    ap.add_argument("--ground-root", type=Path, required=True,
                    help="Root directory for ground truth output")
    ap.add_argument("--datasets", nargs="*", default=None,
                    help="Specific datasets to process (optional)")
    ap.add_argument("--schema-name", default="target",
                    help='DuckDB schema to load tables into (default: "target")')
    
    args = ap.parse_args()
    
    # Find all dataset directories
    ds_dirs = find_dataset_dirs(args.data_root)
    
    # Filter datasets if specified
    if args.datasets:
        ds_dirs = {k: v for k, v in ds_dirs.items() if k in args.datasets}
    
    if not ds_dirs:
        print("[ERROR] No datasets found or specified datasets do not exist")
        sys.exit(1)
    
    print(f"[INFO] Processing {len(ds_dirs)} dataset(s): {list(ds_dirs.keys())}")
    
    # Process each dataset
    for ds_name, ds_dir in ds_dirs.items():
        print(f"\n[INFO] Processing dataset: {ds_name}")
        build_ground_truth_for_dataset(
            ds_name=ds_name,
            ds_dir=ds_dir,
            output_root=args.ground_root,
            schema_name=args.schema_name
        )
    
    print(f"\n[SUCCESS] Ground truth built in: {args.ground_root}")


if __name__ == "__main__":
    try:
        main()
    except duckdb.IOException as e:
        print(f"[ERROR] DuckDB I/O error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
