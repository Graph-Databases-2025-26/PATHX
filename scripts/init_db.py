import duckdb, os

os.makedirs("db", exist_ok=True)
conn = duckdb.connect("db/cust_db.duckdb")
conn.execute("""
CREATE TABLE IF NOT EXISTS customers AS
SELECT * FROM read_csv_auto('data/customers.csv', HEADER=True);
""")
conn.close()

print("Database initialized and customers table created.")