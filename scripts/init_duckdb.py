import duckdb
import pandas as pd
import os

con= duckdb.connect('db/sample.duckdb')

data= pd.read_csv('data/movies/movies.csv')

con.execute("DROP TABLE IF EXISTS movies")
con.execute("""
CREATE TABLE movies (
    primarytitle TEXT,
    originaltitle TEXT,
    startyear INTEGER,
    endyear INTEGER,
    runtimeminutes INTEGER,
    genres TEXT,
    director TEXT,
    birthyear INTEGER,
    deathyear INTEGER
);
""")


con.execute("INSERT INTO movies SELECT * FROM data")

print("✅ DuckDB database created at db/sample.duckdb")
print(con.execute("SELECT * FROM movies LIMIT 5").fetchdf())
