# 🧩 Graph Databases — Main Path Project (PathX)

**University of Padua — A.Y. 2025/26**  
**Course:** Graph Databases  
**Professor:** Gianmaria Silvello  
**Team:** Juliette Tonneau, Minseok Kim, Bita Khashechian, Danil Smirnov

---

# SQL ↔️ LLM Integration (GALOIS-Inspired)

This repository contains the coursework project for **Graph Databases (UniPD)** aimed at integrating a **traditional SQL engine (DuckDB)** with a **Large Language Model (LLM)**, following ideas from the SIGMOD 2025 paper **“Logical and Physical Optimizations for SQL Query Execution over Large Language Models (Galois)”**.

The goal: let users issue **natural-language-like / LLM-backed queries** while still benefiting from **database-style planning, cost/quality trade-offs, and structured outputs**.

---

## 🚩 Objectives

- Treat the **LLM as a data source** (like a table).
- Add **LLM-aware physical operators** (LLM table-scan, key-scan).
- Balance **cost vs. quality** during query planning.
- Keep everything **reproducible and evaluable**: SQL in, JSON out, compare to ground truth.
- Provide a **single setup script** to create the environment, load data, run queries, and export plans.

---

## 📦 Tech Stack

- **Python** 3.11+
- **DuckDB** (local, analytical SQL)
- **IBM watsonx.ai** (LLM calls – later stages)
- **Core libs:** `duckdb`, `pandas`, `sqlglot`, `pydantic`, `python-dotenv`, `pyyaml`
- **Dev:** VS Code

---

## Stage 1 Overview

1. Load CSV / provided data into **DuckDB**.
2. Run SQL queries from `.sql` files.
3. Export results as **JSON** (full: rows + metrics).
4. Compare against **ground truth** for correctness.
5. Keep **EXPLAIN output** for inspection.

Later stages plug in the LLM and replace some scans with **LLMScan** / **Filter-LLMScan**.

---

## 📁 Repository Layout

````text
/data         # datasets, .sql queries, ingestion scripts
/ground       # ground-truth outputs for evaluation
/db           # local DuckDB files (ignored in git)
/scripts      # tutor/evaluation helpers
/src
  /db         # DuckDB init + ingestion
  /llm        # LLM client, prompts, table/key scan
  /pipelines  # run queries, export JSON, baselines
  /utils      # logging, timing, observability
/config
  .env        # secrets (IBM / watsonx)
  config.yaml # model + path + prompt config
/submissions  # final JSONs, EXPLAIN outputs
/test         # unit / integration tests

----

## 🛠️ Installation and Setup

## 1️⃣ Clone the repository

```bash
git clone https://github.com/Graph-Databases-2025-26/PathX.git
cd PathX
````

## 2️⃣ Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate     # macOS / Linux
# or
.venv\Scripts\activate        # Windows
```

## 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

## 4️⃣ Initialize DuckDB and load data

```bash
python scripts/init_duckdb.py
```

This creates db/sample.duckdb and loads data from data/customers.csv.

## ⚙️ Typical usage (manual commands)

```bash
# 1. Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Build the ground truth
python scripts/build_ground_truth.py --data-root data --ground-root ground_truth_result \
  --datasets flight-2 flight-4 fortune geo premier presidents world movies

# 4. To ingest the datasets to database
python src/pipelines/ingest_all.py --datasets <datasetname1> <datasetname2> # by name
python src/pipelines/ingest_all.py # all

# 5. To process queries
python -m src.pipelines.run_queries

# 6. To export EXPLAIN
python src/pipelines/db_explain.py    --datasets <datasetname1> <datasetname2>  --format txt --db-path ./database.duckdb # to .txt
python src/pipelines/db_explain.py    --datasets <datasetname1> <datasetname2> --format json --db-path ./database.duckdb # to .json
#OR
python src/pipelines/db_explain.py    --datasets <datasetname1> <datasetname2> --format both --db-path ./database.duckdb # to .json and .txt

# 7. Verify the evaluation (baseline == ground)
python scripts/galois_eval.py --ground ground --submissions submissions \
  --datasets GEO MOVIES PRESIDENTS WORLD FLIGHT-2 FLIGHT-4 --cell-metric similarity --tuple-metric constraint

  python galois_eval.py --ground ground --submissions submissions \
  --datasets GEO MOVIES PRESIDENTS WORLD FLIGHT-2 FLIGHT-4 \
  --cell-metric similarity --tuple-metric constraint --overall --format table
```
