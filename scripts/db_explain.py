#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SQL EXPLAIN Implementation Module

This module handles:
- Generating query execution plans using EXPLAIN
- Analyzing query performance
- Providing detailed query plan information in JSON format
- Supporting both EXPLAIN and EXPLAIN ANALYZE

Usage:
    python db_explain.py --db-path ./database.duckdb --query "SELECT * FROM target.movies"
    python db_explain.py --db-path ./database.duckdb --query-file ./queries.sql --analyze
"""

import argparse
import json
import sys
from pathlib import Path
import duckdb
import re


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


def explain_query(con: duckdb.DuckDBPyConnection, sql: str, analyze: bool = False):
    """
    Generate execution plan for a SQL query.
    
    Args:
        con: DuckDB connection object
        sql: SQL query string
        analyze: If True, use EXPLAIN ANALYZE (actually executes the query)
        
    Returns:
        Dictionary containing the query plan and metadata
    """
    try:
        # Prepare the EXPLAIN statement
        explain_sql = f"EXPLAIN ANALYZE {sql}" if analyze else f"EXPLAIN {sql}"
        
        # Execute EXPLAIN
        result = con.execute(explain_sql).fetchall()
        
        # Parse the result
        plan_lines = [row[1] if len(row) > 1 else row[0] for row in result]
        plan_text = "\n".join(plan_lines)
        
        # Build result dictionary
        explain_result = {
            "original_query": sql,
            "explain_type": "EXPLAIN ANALYZE" if analyze else "EXPLAIN",
            "plan_text": plan_text,
            "plan_lines": plan_lines,
            "success": True
        }
        
        # If ANALYZE was used, try to extract timing information
        if analyze:
            timing_info = extract_timing_info(plan_text)
            if timing_info:
                explain_result["timing"] = timing_info
        
        return explain_result
        
    except Exception as e:
        return {
            "original_query": sql,
            "explain_type": "EXPLAIN ANALYZE" if analyze else "EXPLAIN",
            "error": str(e),
            "success": False
        }


def extract_timing_info(plan_text: str):
    """
    Extract timing information from EXPLAIN ANALYZE output.
    
    Args:
        plan_text: The execution plan text
        
    Returns:
        Dictionary with timing information or None
    """
    timing_info = {}
    
    # Look for timing patterns in the plan
    # This is a simplified extraction - DuckDB's format may vary
    lines = plan_text.split("\n")
    
    for line in lines:
        # Extract total time if present
        if "Total Time" in line or "Execution Time" in line:
            parts = line.split(":")
            if len(parts) > 1:
                timing_info["total_time"] = parts[1].strip()
    
    return timing_info if timing_info else None


def explain_query_file(db_path: Path, query_file: Path, output_dir: Path = None, analyze: bool = False, export_format: str):
    """
    Generate execution plans for all queries in a SQL file.
    
    Args:
        db_path: Path to the .duckdb database file
        query_file: Path to SQL file containing queries
        output_dir: Directory to save JSON results (optional)
        analyze: If True, use EXPLAIN ANALYZE
        
    Returns:
        Dictionary with explain results for all queries
    """
    con = duckdb.connect(str(db_path), read_only=True)
    
    sqls = read_sql_statements(query_file)
    
    if not sqls:
        print(f"[WARN] No SQL statements found in {query_file}")
        con.close()
        return {}
    
    results = {}
    
    for i, sql in enumerate(sqls, start=1):
        explain_type = "EXPLAIN ANALYZE" if analyze else "EXPLAIN"
        print(f"[INFO] {explain_type} query {i}/{len(sqls)}...")
        
        result = explain_query(con, sql, analyze=analyze)
        results[f"query{i}"] = result
        
        if not result["success"]:
            print(f"[ERROR] Query {i}: {result.get('error', 'Unknown error')}")
            continue

        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

            # JSON export
            if export_format in ("json", "both"):
                json_path = output_dir / f"explain_query{i}.json"
                json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"[SAVED] {json_path}")

            # TXT export
            if export_format in ("txt", "both"):
                txt_path = output_dir / f"explain_query{i}.txt"
                txt_path.write_text(result.get("plan_text", "[No plan available]"), encoding="utf-8")
                print(f"[SAVED] {txt_path}")
        
    
    con.close()
    return results


def explain_single_query(db_path: Path, sql: str, output_dir: Path = None, analyze: bool = False, export_format: str = "json"):
    """
    Generate execution plan for a single SQL query.
    
    Args:
        db_path: Path to the .duckdb database file
        sql: SQL query string
        output_file: Optional path to save JSON result
        analyze: If True, use EXPLAIN ANALYZE
        
    Returns:
        Dictionary with explain result
    """
    con = duckdb.connect(str(db_path), read_only=True)
    
    result = explain_query(con, sql, analyze=analyze)
    
    if result["success"]:
        print(f"[OK] Execution plan generated")
        print("\n=== Query Plan ===")
        print(result["plan_text"])
        
        if analyze and "timing" in result:
            print("\n=== Timing Information ===")
            print(json.dumps(result["timing"], indent=2))
    else:
        print(f"[ERROR] {result.get('error', 'Unknown error')}")
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        if export_format in ("json", "both"):
            json_file = output_dir / "explain_single.json"
            json_file.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(f"[SAVED] {json_file}")

        if export_format in ("txt", "both"):
            txt_file = output_dir / "explain_single.txt"
            txt_file.write_text(result.get("plan_text", "[No plan available]"), encoding="utf-8")
            print(f"[SAVED] {txt_file}")
    
    con.close()
    return result


def compare_queries(db_path: Path, queries: list, output_dir: Path = None):
    """
    Compare execution plans for multiple queries.
    
    Args:
        db_path: Path to the .duckdb database file
        queries: List of SQL query strings
        output_file: Optional path to save comparison results
        
    Returns:
        Dictionary with comparison results
    """
    con = duckdb.connect(str(db_path), read_only=True)
    
    comparison = {
        "query_count": len(queries),
        "queries": []
    }
    
    for i, sql in enumerate(queries, start=1):
        print(f"[INFO] Analyzing query {i}/{len(queries)}...")
        result = explain_query(con, sql, analyze=True)
        comparison["queries"].append({
            "query_number": i,
            "sql": sql,
            "plan": result.get("plan_text", ""),
            "timing": result.get("timing", {}),
            "success": result["success"]
        })
    
    con.close()
    
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        if export_format in ("json", "both"):
            json_file = output_dir / "comparison.json"
            json_file.write_text(
                json.dumps(comparison, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            print(f"[SAVED] {json_file}")

        if export_format in ("txt", "both"):
            txt_file = output_dir / "comparison.txt"
            txt_summary = []
            for q in comparison["queries"]:
                txt_summary.append(f"Query {q['query_number']}:\n{q['plan']}\n")
            txt_file.write_text("\n".join(txt_summary), encoding="utf-8")
            print(f"[SAVED] {txt_file}")
    
    return comparison


def main():
    """Main entry point for EXPLAIN functionality."""
    ap = argparse.ArgumentParser(description="Generate SQL execution plans with EXPLAIN")
    ap.add_argument("--db-path", type=Path, required=True, help="Path to .duckdb database file")
    
    # Query input options
    query_group = ap.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query-file", type=Path, help="Path to SQL file with queries")
    query_group.add_argument("--query", type=str, help="Single SQL query to explain")
    
    # EXPLAIN options
    ap.add_argument("--analyze", action="store_true", 
                    help="Use EXPLAIN ANALYZE (executes query and shows actual timings)")
    
    # Output options
    ap.add_argument("--output-dir", type=Path, help="Directory to save results")
    ap.add_argument("--format", choices=["txt", "json", "both"], default="json",
                    help="Export format: txt, json, or both")
    ap.add_argument("--print-plan", action="store_true", help="Print execution plan to stdout")
    
    args = ap.parse_args()
    
    try:
        if args.query_file:
            # Explain multiple queries from file
            results = explain_query_file(
                db_path=args.db_path,
                query_file=args.query_file,
                output_dir=args.output_dir,
                analyze=args.analyze,
                export_format=args.format
            )
            
            if args.print_plan:
                print("\n=== Execution Plans ===")
                print(json.dumps(results, ensure_ascii=False, indent=2))
                
        else:
            # Explain single query
            result = explain_single_query(
                db_path=args.db_path,
                sql=args.query,
                output_file=args.output_file,
                analyze=args.analyze
            )
            
            if args.print_plan:
                print("\n=== Execution Plan (JSON) ===")
                print(json.dumps(result, ensure_ascii=False, indent=2))
                
    except duckdb.IOException as e:
        print(f"[ERROR] DuckDB I/O error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
