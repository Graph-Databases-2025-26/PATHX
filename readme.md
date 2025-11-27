# PathX: DuckDB and LLM Baselines

Baselines and utilities to reproduce the SIGMOD’25 GALOIS results: build DuckDB databases from CSVs, generate ground truth, run deterministic and LLM-backed SQL/NL baselines, export EXPLAIN/ANALYZE plans, and score submissions with the GALOIS metrics.

Course: University of Padua — Graph Databases 2025/26  
Team: Juliette Tonneau · Minseok Kim · Bita Khashechian · Danil Smirnov

## Quick start
1) Python 3.11+, then:
```bash
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
2) Create `.env` with provider keys (see Config below).  
3) Build ground truth:
```bash
python scripts/build_ground_truth.py --data-root data --ground-root ground --datasets flight-2 flight-4 fortune geo movies premier presidents world
```
4) Ingest datasets and run deterministic SQL:
```bash
python -m src.db_interaction.ingest_all --datasets geo movies
python -m src.db_interaction.run_queries --datasets geo movies
```
5) Run LLM baselines:
```bash
python -m src.baselines.sql_baseline --datasets geo movies --strategy direct --max-iters 5
python -m src.baselines.nl_baseline  --datasets geo movies --strategy direct --max-iters 5
```
6) Evaluate:
```bash
python scripts/galois_eval.py --ground ground --submissions sql_baseline_results --datasets GEO MOVIES --overall --format table
python scripts/galois_eval.py --ground ground --submissions nl_baseline_results  --datasets GEO MOVIES --overall --format table
```

## Repository map
```
data/                  CSVs + queries_*.sql (and optional queries_*.txt, ingest_*.sql)
db/                    DuckDB files created locally (per dataset)
ground/                Ground-truth JSON outputs (query{i}.json)
results/               explain_json/ and explain_txt/ exports
submissions/           Deterministic DuckDB query outputs
sql_baseline_results/  Watsonx SQL baseline outputs (other providers create *_gemini, *_azure)
nl_baseline_results/   Watsonx NL baseline outputs (other providers create *_gemini, *_azure)
config/                AppConfig.yaml + config loader
scripts/               Build ground truth, galois_eval, batch exporters, helpers
src/
  baselines/           LLM baselines (chains, prompts, parsing, output writers)
  db_interaction/      DuckDB ingest/run/explain helpers
  utils/               logging, dataset helpers, EXPLAIN export
reports_pathx/         Coursework reports (not used by code)
requirements.txt
```

## Config and credentials
- Main settings live in `config/AppConfig.yaml`. Environment variables are interpolated (`${VAR}`) when the file is loaded.
- Supported providers: `watsonx`, `gemini`, `azure` (set `llm_provider` in the YAML or override via env `LLM_PROVIDER`).
- Prompt strategies (SQL and NL): `direct`, `zero_shot`, `few_shot`.
- `.env` example:
```
# Watsonx
WATSONX_API_KEY=...
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT_ID=...
WATSONX_MODEL_ID=ibm/granite-3-3-8b-instruct
WATSONX_TEMPERATURE=0.0
WATSONX_MAX_NEW_TOKENS=2048

# Gemini
GEMINI_API_KEY=...
GEMINI_MODEL_NAME="gemini-2.5-flash"

# Azure OpenAI
AZURE_OPENAI_ENDPOINT=...
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_API_VERSION=2024-10-21
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

## Data layout and conventions
- Each dataset lives in `data/<dataset>/` and must include `queries_*.sql`. Optional: `ingest_*.sql` to predefine schema; otherwise CSVs are auto-read.
- NL questions can be provided via `queries_*.txt` using `--query`-style sections (see `src/utils/dataset_io.py` for parsing).
- `flight` is split under `data/flight/flight-2` and `data/flight/flight-4`; the tools expose them as `flight-2` and `flight-4`.

## Pipelines and commands

### Ground truth (reference answers)
Run SQL over CSVs and write `ground/<DATASET>/query{i}.json`:
```bash
python scripts/build_ground_truth.py --data-root data --ground-root ground --datasets geo movies
```

### DuckDB ingestion
Create `db/<DATASET>.duckdb` and load tables into schema `target` (runs `ingest*.sql` if present, else CSV autodetect):
```bash
python -m src.db_interaction.ingest_all --datasets geo movies
```

### Deterministic SQL runner
Execute `queries_*.sql` against DuckDB and store JSON rows under `submissions/<DATASET>/`:
```bash
python -m src.db_interaction.run_queries --datasets geo movies
```

### EXPLAIN / ANALYZE export
Generate text + JSON plans for every query; outputs under `results/explain_txt/` and `results/explain_json/`:
```bash
python -m src.db_interaction.run_explain_all
```

### LLM SQL baseline
- Builds prompt via Jinja2 templates (`src/baselines/prompts/sql/*.j2`) and includes table/schema info from DuckDB.
- Iterative retrieval with follow-up prompts (`max-iters` controls extra “more rows” turns).
- Output path is provider-specific, e.g. `sql_baseline_results/WATSONX/GEO/query1.json`.
```bash
python -m src.baselines.sql_baseline --datasets geo movies --strategy few_shot --max-iters 5
```

### LLM NL baseline
- Reads NL questions from `queries_*.txt` or line-by-line text files.
- Uses the same schema-aware prompting and iterative runner as the SQL baseline.
- Output path is provider-specific, e.g. `nl_baseline_results/WATSONX/GEO/query1.json`.
```bash
python -m src.baselines.nl_baseline --datasets geo movies --strategy direct --max-iters 5
```

### Evaluation (GALOIS metrics)
Compare any submissions folder against ground truth (supports `--cell-metric exact|similarity`, `--tuple-metric exact|similarity`, `--format table|csv|json`):
```bash
python scripts/galois_eval.py --ground ground --submissions sql_baseline_results --datasets GEO MOVIES --overall --format table
```

### Paper comparison helper
Runs the evaluator for both baselines and prints deltas vs published numbers:
```bash
python -m src.baselines.compare_results
```

### Batch exporter for py-galois (optional)
Calls a running py-galois server and saves outputs under `submissions/<STUDENT>/<DATASET>/`:
```bash
python scripts/run_batch_export.py --data-root data --schemas-root schemas --submissions-root submissions --student myname --datasets geo --api-url http://127.0.0.1:8000/query
```

## Output folders at a glance
- `db/<DATASET>.duckdb`        DuckDB snapshot per dataset (schema `target`)
- `ground/<DATASET>/query*.json`   Reference answers
- `submissions/<DATASET>/query*.json`   Deterministic DuckDB runs
- `sql_baseline_results/*/<DATASET>/query*.json`   LLM SQL baseline (provider-specific root)
- `nl_baseline_results/*/<DATASET>/query*.json`    LLM NL baseline (provider-specific root)
- `results/explain_{txt,json}/<DATASET>/`          EXPLAIN and EXPLAIN ANALYZE dumps

## Notes
- Make sure DuckDB files are built (ingest step) before running LLM baselines so schema introspection succeeds.
- `config/AppConfig.yaml` pulls values from `.env`; missing keys will raise at startup.
- JSON outputs include `result_set`, elapsed `time`, total `tokens`, and `meta.iterations_used` for iterative runs.
- Keep provider secrets out of version control; `.env` is gitignored by default.
