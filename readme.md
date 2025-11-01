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

```
----

### 🛠️ Installation and Setup

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
# 🛠️ Commands and functions

## 🧭 Full Automated Pipeline
Run everything end-to-end with a single command 👇
```bash
python scripts/run_all.py
```

Options:
```bash
python scripts/run_all.py --datasets movies world        # subset
python scripts/run_all.py --skip-explain                 # skip EXPLAIN
python scripts/run_all.py --baseline                     # compare ground vs ground
```
## 🧩 Manually:

## ⚙️ Typical usage (manual commands)

```bash
python scripts/build_ground_truth.py --data-root data --ground-root ground \
  --datasets flight-2 flight-4 fortune geo premier presidents world movies

## 2. To ingest the datasets to database
```bash
python src/db/ingest_all.py --datasets <datasetname1> <datasetname2> # by name
python src/db/ingest_all.py # all
```

# 5. To process queries
python -m src.pipelines.run_queries
```

## 4. To export EXPLAIN
```bash
python -m src.pipelines.run_explain_all
```
Results are stored under:
```bash
results/
├─ explain_json/<DATASET>/queryN__explain.json
├─ explain_json/<DATASET>/queryN__analyze.json
├─ explain_txt/<DATASET>/queryN__explain.txt
└─ explain_txt/<DATASET>/queryN__analyze.txt
```

## 5. Verify the evaluation (baseline == ground)
```bash
python scripts/galois_eval.py --ground ground --submissions submissions \
  --datasets GEO MOVIES PRESIDENTS WORLD FLIGHT-2 FLIGHT-4 --cell-metric similarity --tuple-metric constraint
```
or formatted output:
```bash
  python scripts/galois_eval.py --ground ground --submissions submissions \
  --datasets GEO MOVIES PRESIDENTS WORLD FLIGHT-2 FLIGHT-4 \
  --cell-metric similarity --tuple-metric constraint --overall --format table
```

# 📂 Project Structure
```bash
pathX/
├── data/                      # all input datasets (CSV, SQL, JSON)
├── db/                        # generated DuckDB databases per dataset
├── ground_truth_result/       # reference query outputs
├── results/
│   ├── explain_json/          # JSON plans (EXPLAIN / ANALYZE)
│   └── explain_txt/           # text-based EXPLAIN trees
├── submissions/               # generated query results
│
├── scripts/                   # CLI scripts and automation tools
│   ├── build_ground_truth.py
│   ├── galois_eval.py
│   ├── run_all.py
│   └── ...
│
├── src/
│   ├── db/                    # database clients (DuckDB)
│   ├── pipelines/             # ingestion, queries, explain pipelines
│   ├── utils/                 # logging, explain_export, etc.
│   └── ...
│
└── requirements.txt
```