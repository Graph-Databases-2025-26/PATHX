#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from pathlib import Path

from db_init import find_dataset_dirs, initialize_database
from db_query import read_sql_statements, execute_single_query
from db_explain import explain_single_query
import duckdb

def run_full_pipeline(data_root: Path, output_root: Path, datasets=None, schema_name="target"):
    db_dir = output_root / "database"
    results_dir = output_root / "results"
    explain_dir = output_root / "explain"
    db_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    explain_dir.mkdir(parents=True, exist_ok=True)
    db_path = db_dir / "database.duckdb"

    stats = initialize_database(
        data_root=data_root,
        db_path=db_path,
        schema_name=schema_name,
        datasets=datasets
    )

    ds_dirs = find_dataset_dirs(data_root)
    if datasets:
        ds_dirs = {k: v for k, v in ds_dirs.items() if k in datasets}
    if not ds_dirs:
        return

    for ds_name, ds_dir in ds_dirs.items():
        qfiles = list(ds_dir.glob("queries_*.sql"))
        if not qfiles:
            continue
        qpath = sorted(qfiles)[0]
        sqls = read_sql_statements(qpath)
        if not sqls:
            continue

        ds_results_dir = results_dir / ds_name.upper()
        ds_explain_dir = explain_dir / ds_name.upper()
        ds_results_dir.mkdir(parents=True, exist_ok=True)
        ds_explain_dir.mkdir(parents=True, exist_ok=True)

        for i, sql in enumerate(sqls, start=1):
            result_file = ds_results_dir / f"query{i}.json"
            try:
                execute_single_query(
                    db_path=db_path,
                    sql=sql,
                    output_file=result_file
                )
            except Exception:
                pass

        for i, sql in enumerate(sqls, start=1):
            explain_file = ds_explain_dir / f"explain_query{i}.json"
            try:
                explain_single_query(
                    db_path=db_path,
                    sql=sql,
                    output_file=explain_file,
                    analyze=True
                )
            except Exception:
                pass

def main():
    ap = argparse.ArgumentParser(
        description="Pipeline complet, mode silencieux"
    )
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--schema-name", default="target")
    args = ap.parse_args()
    try:
        run_full_pipeline(
            data_root=args.data_root,
            output_root=args.output_root,
            datasets=args.datasets,
            schema_name=args.schema_name
        )
    except Exception:
        sys.exit(1)

if __name__ == "__main__":
    main()
