#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Query Execution and Results Module

This module handles:
- Reading SQL query files
- Executing queries against DuckDB database
- Returning results in JSON format
- Batch processing of multiple queries

Usage:
    python db_query.py --db-path ./database.duckdb --query-file ./queries.sql --output-dir ./results
    python db_query.py --db-path ./database.duckdb --query "SELECT * FROM target.movies LIMIT 10"
"""

import argparse
import json
import re
import sys
from pathlib import Path
import duckdb
import pandas as pd


def read_sql_statements(path: Path):
    """
    Read and parse SQL statements from a file.
    
    Args:
        path: Path to the SQL file
        
    Returns:
        List of SQL statement strings
    """
    text = path.read_text(encoding="utf-8")
    
    # Strip simple -- comments
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("--"):
            continue
        lines.append(line)
    text = "\n".join(lines)
    
    # Split by semicolons that terminate statements
    # Keep robust split: newlines + ; + optional trailing comment
    parts = [p.strip() for p in re.split(r';\s*(?:--.*)?\n', text) if p.strip()]
    
    stmts = []
    for p in parts:
        for q in [s.strip() for s in p.split(';') if s.strip()]:
            if q:
                stmts.append(q)
    
    # Deduplicate preserving order
    out, seen = [], set()
    for s in stmts:
        if s not in seen:
            seen.add(s)
            out.append(s)
    
    return out


def execute_query(con: duckdb.DuckDBPyConnection, sql: str, return_format="json"):
    """
    Execute a single SQL query and return results.
    
    Args:
        con: DuckDB connection object
        sql: SQL query string to execute
        return_format: Output format - "json", "dict", or "dataframe"
        
    Returns:
        Query results in the specified format
    """
    try:
        df = con.execute(sql).df()
        
        if return_format == "dataframe":
            return df
        elif return_format == "dict":
            return df.to_dict(orient="records")
        else:  # json
            rows = df.to_dict(orient="records")
            return json.dumps(rows, ensure_ascii=False, indent=2)
            
    except Exception as e:
        error_msg = f"Query execution error: {str(e)}"
        if return_format == "json":
            return json.dumps({"error": error_msg}, indent=2)
        else:
            return {"error": error_msg}


def execute_query_file(db_path: Path, query_file: Path, output_dir: Path = None, schema_name: str = "target"):
    """
    Execute all queries from a SQL file and save results.
    
    Args:
        db_path: Path to the .duckdb database file
        query_file: Path to SQL file containing queries
        output_dir: Directory to save JSON results (optional)
        schema_name: Schema name used in the database
        
    Returns:
        Dictionary with query results and statistics
    """
    con = duckdb.connect(str(db_path), read_only=True)
    
    sqls = read_sql_statements(query_file)
    
    if not sqls:
        print(f"[WARN] No SQL statements found in {query_file}")
        con.close()
        return {}
    
    results = {}
    
    for i, sql in enumerate(sqls, start=1):
        print(f"[INFO] Executing query {i}/{len(sqls)}...")
        
        try:
            df = con.execute(sql).df()
            rows = df.to_dict(orient="records")
            results[f"query{i}"] = {
                "sql": sql,
                "row_count": len(rows),
                "results": rows
            }
            
            # Save to file if output directory is specified
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"query{i}.json"
                output_file.write_text(
                    json.dumps(rows, ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
                print(f"[OK] Query {i}: {len(rows)} rows -> {output_file}")
            else:
                print(f"[OK] Query {i}: {len(rows)} rows")
                
        except Exception as e:
            print(f"[ERROR] Query {i}: {e}")
            results[f"query{i}"] = {
                "sql": sql,
                "error": str(e),
                "row_count": 0,
                "results": []
            }
            
            if output_dir:
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / f"query{i}.json"
                output_file.write_text(
                    json.dumps([], ensure_ascii=False, indent=2),
                    encoding="utf-8"
                )
    
    con.close()
    return results


def execute_single_query(db_path: Path, sql: str, output_file: Path = None):
    """
    Execute a single SQL query and optionally save results.
    
    Args:
        db_path: Path to the .duckdb database file
        sql: SQL query string
        output_file: Optional path to save JSON results
        
    Returns:
        Query results as dictionary
    """
    con = duckdb.connect(str(db_path), read_only=True)
    
    try:
        df = con.execute(sql).df()
        rows = df.to_dict(orient="records")
        
        result = {
            "sql": sql,
            "row_count": len(rows),
            "results": rows
        }
        
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(
                json.dumps(rows, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(f"[OK] Results saved to: {output_file}")
        
        print(f"[OK] Query executed: {len(rows)} rows returned")
        return result
        
    except Exception as e:
        print(f"[ERROR] {e}")
        return {
            "sql": sql,
            "error": str(e),
            "row_count": 0,
            "results": []
        }
    finally:
        con.close()


def main():
    """Main entry point for query execution."""
    ap = argparse.ArgumentParser(description="Execute SQL queries against DuckDB database")
    ap.add_argument("--db-path", type=Path, required=True, help="Path to .duckdb database file")
    
    # Query input options
    query_group = ap.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query-file", type=Path, help="Path to SQL file with queries")
    query_group.add_argument("--query", type=str, help="Single SQL query to execute")
    
    # Output options
    ap.add_argument("--output-dir", type=Path, help="Directory to save JSON results")
    ap.add_argument("--output-file", type=Path, help="File to save single query result")
    ap.add_argument("--schema-name", default="target", help='Schema name (default: "target")')
    ap.add_argument("--print-results", action="store_true", help="Print results to stdout")
    
    args = ap.parse_args()
    
    try:
        if args.query_file:
            # Execute multiple queries from file
            results = execute_query_file(
                db_path=args.db_path,
                query_file=args.query_file,
                output_dir=args.output_dir,
                schema_name=args.schema_name
            )
            
            if args.print_results:
                print("\n=== Query Results ===")
                print(json.dumps(results, ensure_ascii=False, indent=2))
                
        else:
            # Execute single query
            result = execute_single_query(
                db_path=args.db_path,
                sql=args.query,
                output_file=args.output_file
            )
            
            if args.print_results:
                print("\n=== Query Result ===")
                print(json.dumps(result, ensure_ascii=False, indent=2))
                
    except duckdb.IOException as e:
        print(f"[ERROR] DuckDB I/O error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
