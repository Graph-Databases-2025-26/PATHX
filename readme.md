# 🧩 Graph Databases — Main Path Project (PathX)

**University of Padua — A.Y. 2025/26**  
**Course:** Graph Databases  
**Professor:** Gianmaria Silvello  
**Team:** Juliette Tonneau, Minseok Kim, Bita Khashechian, Danil Smirnov

---

## 📘 Project Description

This repository contains the implementation of the **Main Path** project for the _Graph Databases_ course.

The goal is to build a **system that integrates an SQL database** with **a Large Language Model (LLM)**, enabling natural language query processing and query optimization inspired by the **Galois** paper presented in the course.

---

## ⚙️ Stage 1 — Tech Selection and System Setup

This stage focuses on:

- selecting and setting up the development environment,
- preparing a local SQL database (DuckDB),
- implementing a Python pipeline that:
  1. connects to the database,
  2. runs SQL queries from a `.sql` file,
  3. saves results in a `.json` output file in Full format (rows + metrics) defined by the tutor.
  4. assess our system’s outputs by comparing them with the ground-truth data

> The teacher must be able to execute a single Python command to generate a JSON with result sets for each query.  
> _(Reference: Table 2 of the paper.)_

---

## GALOIS Paper: overview and key insights

The SIGMOD 2025 paper _“Logical and Physical Optimizations for SQL Query Execution over Large Language Models”_ presents Galois, a system that enables SQL query execution directly over Large Language Models (LLMs). The observation is that, although LLMs are capable of answering declarative queries and extracting structured data from unstructured sources, their performance declines with complex queries, often resulting in incomplete or inaccurate outputs. Existing database optimization techniques target first cost or latency reduction without addressing adequately the challenge of ensuring result quality.

Galois implements an intermediate optimization layer between SQL or NL queries and LLMs while treating the LLM as a storage layer. The system modifies both logical and physical query plans to adapt to the distinct operational characteristics of language models.

Galois presents several key innovations. First, it redefines the query optimization process to balance execution cost and result quality. It also introduces two physical operators for retrieving data from LLMs: table-scan and key-scan. Since LLMs lack catalogs, Galois dynamically builds metadata and estimates confidence scores for query execution based on the model’s self-assessed certainty.

The main lessons we keep in mind for our implementation are:

- Introduce logical and physical plans that adapt query decomposition to the LLM's strengths and limitations
- Treat the LLM as a data source, not just a reasoning tool
- Maintain a cost-quality balance
- Structure results in JSON to ensure compatibility with analysis tools

---

## Detailed Implementation schedule

**Phase 1 – Tech Selection and System Setup:**
We’ll begin by setting up our full system — including the architecture design, Python environment, and DuckDB databases — with all setup tests and documentation done.

**Phase 2 – NL and SQL Baselines:**
we’ll build our natural language and SQL baselines, integrate them into LangChain, test different prompting strategies, and set up evaluation metrics to measure accuracy and F1 scores.

**Phase 3 – Implementation of Galois Features (Part I):**
In this phase we’ll implement the core Galois features like LLMScan, Filter-LLMScan, joins, and two-phase retrieval, leading up to a demo comparing naïve and optimized runs.

**Phase 4 – Implementation of Galois Features (Part II):**
Then we’ll develop and test the optimized versions of Galois (Selective, All-Attribute, and Full), replicate key results, and finalize our report.

---

## Technology Stack

| Component       | Tool / Library                                 | Description                                                  |
| --------------- | ---------------------------------------------- | ------------------------------------------------------------ |
| **Language**    | Python 3.11+                                   | Main programming language                                    |
| **Database**    | DuckDB                                         | Lightweight, file-based SQL database optimized for analytics |
| **LLM (later)** | IBM WatsonX.ai / OpenAI API                    | Large Language Model integration for later stages            |
| **Libraries**   | `duckdb`, `pandas`, `sqlglot`, `python-dotenv` | Core Python dependencies                                     |
| **IDE**         | Visual Studio Code                             | Main development environment                                 |

---

## 🛠️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Graph-Databases-2025-26/PathX.git
cd PathX
```

## 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate     # macOS / Linux
# or
.venv\Scripts\activate        # Windows
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```
## 🛠️ Commands and functions

### 🧭 Full Automated Pipeline
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
### 🧩 Manually:

#### 1. Build the ground truth
```bash
python scripts/build_ground_truth.py --data-root data --ground-root ground \
  --datasets flight-2 flight-4 fortune geo premier presidents world movies
```

#### 2. To ingest the datasets to database
```bash
python src/db/ingest_all.py --datasets <datasetname1> <datasetname2> # by name
python src/db/ingest_all.py # all
```

#### 3. To process queries
```bash
python -m src.pipelines.run_queries
```

#### 4. To export EXPLAIN
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

#### 5. Verify the evaluation (baseline == ground)
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

## 📂 Project Structure
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