# GALOIS: LLM-Driven Query Processing Framework
*University of Padua — Graph Databases 2025/26*  
**Team PATHX:** Juliette Tonneau · Minseok Kim · Bita Khashechian · Danil Smirnov

This repository contains two major components:

# Deterministic Pipelines
Full DuckDB-based pipeline used to reproduce baseline results from the SIGMOD’25 *GALOIS* paper:  
- build ground truth  
- ingest CSV datasets  
- run deterministic SQL baselines  
- run LLM SQL and NL baselines (WatsonX, Gemini, Azure)  
- evaluate with GALOIS metrics  
- export EXPLAIN/ANALYZE plans

# GALOIS Engine 
A complete and extensible re-implementation of the *GALOIS* LLM query processor (2025):

- Table-Scan (Algorithm 1)  
- Key-Scan (Algorithm 2)  
- GALOIS-A (Push-All)  
- GALOIS-S (Selective Pushdown)  
- GALOIS-F (Full Optimization: logical + physical)  
- GALOIS-WO (cell-by-cell worst-case baseline)

This module can execute SQL queries directly via LLM operators, compare algorithms, and produce full cost/time/token analyses.

# Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

# Configure LLM Providers

Create `.env` with your keys:

```
# WatsonX
WATSONX_API_KEY=...
WATSONX_URL=...
WATSONX_PROJECT_ID=...
WATSONX_MODEL_ID=meta/llama-3-3-70b-instruct

# Gemini
GEMINI_API_KEY=...
GEMINI_MODEL_NAME="gemini-2.5-flash"

# Azure OpenAI
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

# Repository Structure

```
src/
├── baselines/
│   ├── sql_baseline.py
│   ├── nl_baseline.py
│   └── prompts/
│
├── galois/
│   ├── core/
│   ├── operators/
│   ├── prompts/
│   ├── runner/
│   └── README_galois.md
│
├── db_interaction/
├── config/
├── utils/
├── scripts/
└── test
```

# Part I — Pipelines

### Build ground truth
```bash
python -m scripts.build_ground_truth --data-root data --ground-root ground --datasets geo movies
```

### Ingest datasets
```bash
python -m src.db_interaction.ingest_all --datasets geo movies
```

### Run SQL Baseline
```bash
python -m src.baselines.sql_baseline --datasets geo movies --strategy direct
```

### Run NL Baseline
```bash
python -m src.baselines.nl_baseline --datasets geo movies --strategy direct
```
### Evaluate
```bash
python -m scripts.galois_eval --ground ground --submissions sql_baseline_results --overall
```

### Comparison
```bash
python -m src.baselines.compare_results
```

# Part II — GALOIS LLM Query Engine

Run full GALOIS suite:

```bash
python -m src.galois.run_galois \
  --datasets GEO MOVIES PRESIDENTS WORLD FLIGHT-2 FLIGHT-4 \
  --variants galois_wo galois_s galois_a galois_f
```

# GALOIS Algorithms

- **Table-Scan** – retrieve full rows iteratively  
- **Key-Scan** – retrieve keys then tuples  
- **GALOIS-A** – push all predicates  
- **GALOIS-S** – selective pushdown via LLM-estimated selectivity  
- **GALOIS-F** – full logical + physical optimization  
- **GALOIS-WO** – cell-by-cell worst case baseline  

---

# Prompt Layer (Jinja2)

Prompts are stored in:

```
src/galois/prompts/templates/*.j2
```

Loaded via:

```python
from jinja2 import Environment, FileSystemLoader
env = Environment(loader=FileSystemLoader("src/galois/prompts/templates"))
prompt = env.get_template("table_scan_first.j2").render(...)
```

---

# Sample Output

```
+-----------+--------+----------+------------+--------------+
| Variant   | Rows   | Tokens   | Time (s)   | Strategy     |
|-----------+--------+----------+------------+--------------|
| Galois_WO | 5      | 11648    | 28.60      | cell_by_cell |
| Galois_A  | 4      | 597      | 2.98       | table_scan   |
| Galois_S  | 5      | 848      | 3.58       | table_scan   |
| Galois_F  | 8      | 4043     | 20.21      | key_scan     |
+-----------+--------+----------+------------+--------------+
```

---

# Output folders

```
nl_baseline_results/<dataset>/
sql_baseline_results/<dataset>/
galois_results_wo/<dataset>/
galois_results_a/<dataset>/
galois_results_f/<dataset>/
galois_results_s/<dataset>/
```

---
