import duckdb, json

def run_queries(db_path, sql_file, out_json):
    # Connect to the DuckDB database
    conn = duckdb.connect(db_path)

    # Read SQL queries from the file
    with open(sql_file) as f:
        queries = [q.strip() for q in f.read().split(';') if q.strip()]
    results = {}
    for i, q in enumerate(queries, 1):
        results[f"query_{i}"] = conn.execute(q).fetchall()
    # Close the connection
    conn.close()

    # Write results to JSON file
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Query results written to {out_json}")

if __name__ == "__main__":
    run_queries("db/cust_db.duckdb", "queries/queries.sql", "results.json")