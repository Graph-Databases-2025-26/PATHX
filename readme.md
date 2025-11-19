# 🧩 PathX — Graph Databases + LLM Baselines

**University of Padua · Graph Databases 2025/26**  
Team: Juliette Tonneau · Minseok Kim · Bita Khashechian · Danil Smirnov

This repository reproduces the SQL↔LLM experiments from the SIGMOD’25 *GALOIS* paper.  
The codebase lets you: ingest datasets into DuckDB, build ground-truth answers, run LLM-powered SQL/NL baselines (Azure OpenAI, Google Gemini, IBM watsonx), export EXPLAIN plans, and compare your runs against the paper metrics.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Layout](#repository-layout)
3. [Prerequisites](#prerequisites)
4. [Environment & Data Setup](#environment--data-setup)
5. [Core Pipelines](#core-pipelines)
6. [Baseline Orchestration & Evaluation](#baseline-orchestration--evaluation)
7. [LLM Providers & Prompts](#llm-providers--prompts)
8. [Utilities & Shortcuts](#utilities--shortcuts)
9. [Development Notes](#development-notes)

---

## Project Overview

| Stage | Description | Key Outputs |
|-------|-------------|-------------|
| Ground Truth | Execute queries over CSVs with DuckDB to create reference answers. | `ground/<DATASET>/queryN.json` |
| DuckDB Prep | Build per-dataset `.duckdb` files for reuse (queries, EXPLAIN). | `db/<DATASET>.duckdb` |
| Deterministic SQL | Run queries directly on DuckDB to simulate perfect execution. | `submissions/<DATASET>/queryN.json` |
| NL & SQL LLM Baselines | Prompt LLMs to answer SQL (structured output) or NL-style questions. | `submissions/`, `sql_baseline_results/` |
| Evaluation | Compare any submission folder against ground truth with GALOIS metrics. | CLI tables/CSV + comparison tables |
| Explain Export | Save EXPLAIN/ANALYZE text + JSON for each query. | `results/explain_*` |

The repository is intentionally modular so every step can be repeated independently or wired together through helper scripts.

---

## Repository Layout

```
pathX/
├── data/                     # datasets (CSV + queries_*.sql) – you provide
├── db/                       # DuckDB files, created locally
├── ground/                   # ground-truth query outputs
├── results/                  # explain_json / explain_txt
├── submissions/              # deterministic (DuckDB) query outputs
├── sql_baseline_results/     # LLM SQL baseline outputs
├── scripts/                  # automation helpers & evaluation tools
├── src/
│   ├── db_interaction/       # DuckDB ingest helpers
│   ├── llm/                  # provider clients + prompts
│   ├── pipelines/            # CLI pipelines (queries, baselines, compare)
│   ├── utils/                # logging, timers, JSON helper
│   └── ...
├── requirements.txt
└── readme.md
```

---

## Prerequisites

- **Python** ≥ 3.11 (virtual environment recommended).
- **DuckDB** (Python package installs automatically).
- `pip install -r requirements.txt`
- Access credentials for at least one LLM provider (IBM watsonx, Google Gemini, or Azure OpenAI).
- Datasets under `data/` matching the official PathX layout (CSV + `queries_*.sql`, optional `ingest_*.sql`).

---

## Environment & Data Setup

### 1. Create & activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Linux/macOS
# .venv\Scripts\activate       # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Configure environment variables

Create `.env` (or export vars) with the credentials you intend to use:

```
# IBM watsonx
WATSONX_API_KEY=...
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT_ID=...
WATSONX_MODEL_ID=ibm/granite-3-3-8b-instruct

# Google Gemini
GEMINI_API_KEY=...
GEMINI_MODEL=gemini-2.5-flash

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_DEPLOYMENT=gpt-4o

# Optional default provider override
LLM_PROVIDER=watsonx
```

The provider modules read from `.env` via `python-dotenv`. Keep real keys out of version control.

### 3. Place datasets

```
data/
├── movies/
│   ├── *.csv
│   └── queries_*.sql
├── flight/
│   ├── flight-2/...
│   └── flight-4/...
└── ...
```

Optional `ingest_*.sql` files are executed before CSV fallbacks, so you can predefine custom schemas.

---

## Core Pipelines

### Build ground truth (JSON answers)

```bash
python scripts/build_ground_truth.py \
  --data-root data \
  --ground-root ground \
  --datasets flight-2 flight-4 fortune geo movies premier presidents world
```

This loads each dataset into an ad-hoc DuckDB instance and executes every query in `queries_*.sql`.  
Outputs land in `ground/<DATASET>/queryN.json`.

### Ingest datasets into DuckDB files

```bash
python src/db_interaction/ingest_all.py --datasets geo movies
```

Creates `db/<DATASET>.duckdb` (one per dataset), running `ingest*.sql` when present or auto-creating tables from CSVs.  
Used later by the deterministic SQL pipeline and EXPLAIN exporter.

### Deterministic SQL execution (baseline 0)

```bash
python -m src.pipelines.run_queries --datasets movies geo
```

Runs the queries directly inside DuckDB and writes canonical JSON answers to `submissions/<DATASET>/queryN.json`. These files can be compared against ground truth to verify ingestion fidelity.

### Export EXPLAIN / ANALYZE plans

```bash
python -m src.pipelines.run_explain_all
```

For each dataset/query, saves text and JSON explain plans under `results/explain_txt` and `results/explain_json`.

---

## Baseline Orchestration & Evaluation

### 1. SQL baseline (LLM executes SQL and must return JSON rows)

```
python -m src.pipelines.sql_baseline \
  --datasets movies geo \
  --strategy fewshot \
  --provider watsonx
```

- **Strategies**: `strict_json`, `json_example`, `constrained`, `hidden_cot`, `fewshot`, `compact`.  
- Outputs stored in `sql_baseline_results/<DATASET>/queryN.json` with metadata (`time`, `tokens`, `result_set`).

### 2. NL baseline (LLM answers natural-language prompt derived from SQL)

```
python -m src.pipelines.nl_baseline \
  --datasets movies geo \
  --provider gemini
```

Uses the prompt template at `src/llm/prompts/nl_query.txt`. Results are written to `submissions/<DATASET>/queryN.json` (same schema as deterministic runs).

### 3. Compare against paper metrics

After both baselines produce outputs, evaluate them with the optimized GALOIS script:

```
python scripts/galois_eval.py \
  --ground ground \
  --submissions sql_baseline_results \
  --datasets GEO MOVIES \
  --cell-metric similarity \
  --tuple-metric similarity \
  --overall --format table
```

For a consolidated summary versus the paper numbers use:

```
python -m src.pipelines.compare_results
```

This runs the evaluator for NL + SQL folders and prints a side-by-side table (ours vs paper, absolute/relative deltas).

### 4. All-in-one baseline runner

```
python scripts/run_baselines_and_compare.py \
  --datasets geo movies \
  --nl-provider watsonx \
  --sql-provider gemini \
  --sql-strategy strict_json
```

Flags allow skipping NL/SQL/compare steps or running compare-only.

### 5. Full end-to-end pipeline

`scripts/generate_ground_truth.py` wraps every stage (build ground, ingest, run queries, explain, evaluate).  
Example:

```
python scripts/generate_ground_truth.py \
  --datasets flight-2 flight-4 movies \
  --skip-explain
```

- `--baseline` evaluates ground vs ground (sanity check).
- Use `--skip-*` flags to resume partially completed runs.

---

## LLM Providers & Prompts

`src/llm/` contains one client per provider:

| Provider | Module | Notes |
|----------|--------|-------|
| IBM watsonx | `watsonx_client.py` | Uses `ibm_watsonx_ai.ModelInference`. Returns `(text, total_tokens)` |
| Google Gemini | `gemini_client.py` | Wraps `google.generativeai` and extracts text/usage. |
| Azure OpenAI | `azure_client.py` | Chat completions API; handles streaming content parts. |

Set `LLM_PROVIDER` env var (or CLI `--provider`) to choose the backend at runtime.  
Prompt engineering lives alongside the pipelines:

- SQL baseline: prompts embedded in `src/pipelines/sql_baseline.py` (strategy-dependent).  
- NL baseline: edit `src/llm/prompts/nl_query.txt` to tweak instructions globally.

Every provider callable must expose `generate_text(prompt) -> tuple[str, int | None]`, making it easy to add new backends.

---

## Utilities & Shortcuts

- `scripts/run_batch_export.py`: calls a running **py-galois** service via HTTP and stores JSON outputs plus metadata (`queryN.meta.json`). Handy for large-scale evaluations.
- `scripts/run_baselines_and_compare.py`: orchestrates NL baseline → SQL baseline → comparison in one command.
- `src/pipelines/galois_pipeline.py`: placeholder entry point for experimenting with custom LLM-aware physical plans (LLMScan, Filter-LLMScan, etc.).
- `src/utils/`: shared helpers for I/O, timers, logging, and explain export logic.

---

## Development Notes

- Keep results and secrets out of git: add `.env`, credentials, or huge dataset dumps to `.gitignore`.
- Prefer `python -m <module>` when invoking pipelines; relative imports assume repo root as working directory.
- When editing prompts or adding providers, ensure outputs remain valid JSON arrays of row objects (`result_set`). The evaluator relies on that schema.
- Testing tips:
  1. Start with deterministic runs (`run_queries`) and compare to ground truth using `scripts/galois_eval.py --submissions submissions`.
  2. Once ingestion matches, switch to LLM baselines and compare via `sql_baseline_results` / `submissions`.
- Logs default to INFO; adjust via `LOGLEVEL=DEBUG python -m ...` if needed.

Questions, improvements, or bug reports are welcome—open an issue or reach out on the course channels. Happy hacking! 🎯
