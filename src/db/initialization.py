from pathlib import Path
from duckdb_client import connect_dataset_db, ingest_dataset

con = connect_dataset_db("flight-4", persist=True)
ingest_dataset(con, Path("data/flight-4"))

print(con.execute("SHOW TABLES FROM target").fetchall())

con.close()
