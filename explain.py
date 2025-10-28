import duckdb
from pathlib import Path

con = duckdb.connect('data/world/worldE.duckdb')  # یا هر پایگاه‌داده‌ای که داری
# اگر لازم است قبلش ingest را اجرا کن: con.execute(open('ingest_world.sql').read())

queries = [
    "SELECT count(distinct government_form) FROM target.country WHERE continent='Africa'",
    "SELECT name FROM target.country WHERE TRY_CAST(independence_year AS INTEGER) > 1950",
    "SELECT avg(gnp), sum(population) FROM target.country WHERE government_form='US Territory'",
    "SELECT distinct t2.region FROM target.country_language t1 JOIN target.country t2 ON t1.country_code_3_letters=t2.code_3_letters WHERE t1.language = 'English' or t1.language = 'Dutch';"
]

out = []
for i, q in enumerate(queries, 1):
    rows = con.execute(f"EXPLAIN {q}").fetchall()
    text = "\n".join(r[0] for r in rows)
    out.append(f"-- Q{i}\n{text}\n")

Path("queries_world_explain.txt").write_text("\n".join(out), encoding="utf-8")
print("Saved: queries_world_explain.txt")