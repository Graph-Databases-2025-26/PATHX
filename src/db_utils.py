import duckdb

def connect_db(dataset: str):
    return duckdb.connect(f"db/{dataset.upper()}.duckdb")

def run_query(con, sql: str):
    cur = con.execute(sql)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]  # column names
    # Convert into the list of objects (simple)
    result = [dict(zip(cols, r)) for r in rows]
    return result
