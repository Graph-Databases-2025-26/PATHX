from pathlib import Path
import sys

from .duckdb_client import connect_dataset_db, ingest_dataset
from ..utils.explain_export import save_both
from ..utils.dataset_io import discover_datasets, read_queries
from ..utils.logging import logger


def process_dataset(dataset: str):
    data_dir = Path("data") / dataset
    db_path = Path("db") / f"{dataset.upper()}.duckdb"

    # 1. Create or reuse database
    con = connect_dataset_db(dataset, persist=True)
    ingest_dataset(con, data_dir, schema="target")
    con.close()

    logger.info(f"[OK] {dataset}: database initialized → {db_path}")

    # 2. Execute EXPLAIN for each query
    queries = read_queries(data_dir)
    if not queries:
        logger.warning(f"[WARN] No queries found for {dataset}")
        return

    for i, sql in enumerate(queries, start=1):
        out_base = Path(f"{dataset}/query{i}")
        logger.info(f"[{dataset}] Running EXPLAINs for query {i}/{len(queries)}")
        save_both(db_path, sql, out_base)


def main():
    datasets = discover_datasets()
    logger.info(f"[INFO] Datasets: {', '.join(datasets)}")

    for ds in datasets:
        try:
            process_dataset(ds)
        except Exception as e:
            logger.error(f"[ERROR] Failed on {ds}: {e}")


if __name__ == "__main__":
    main()
