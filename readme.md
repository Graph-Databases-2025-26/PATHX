# 🧩 Graph Databases — Main Path Project (PathX)

**University of Padua — A.Y. 2025/26**  
**Course:** Graph Databases  
**Professor:** Gianmaria Silvello  
**Team:** Juliette Tonneau, Minseok Kim, Bita Khashechian, Danil Smirnov

---

## 📘 Project Description

This repository contains the implementation of the **Main Path** project for the *Graph Databases* course.

The goal is to build a **system that integrates an SQL database** with **a Large Language Model (LLM)**, enabling natural language query processing and query optimization inspired by the **Galois** paper presented in the course.

---

## ⚙️ Stage 1 — Tech Selection and System Setup

This stage focuses on:
- selecting and setting up the development environment,
- preparing a local SQL database (DuckDB),
- implementing a Python pipeline that:
  1. connects to the database,
  2. runs SQL queries from a `.sql` file,
  3. saves results in a `.json` output file.

> The teacher must be able to execute a single Python command to generate a JSON with result sets for each query.  
> *(Reference: Table 2 of the paper.)*

---

## GALOIS Paper: overview and key insights

The SIGMOD 2025 paper *“Logical and Physical Optimizations for SQL Query Execution over Large Language Models”* presents Galois, a system that enables SQL query execution directly over Large Language Models (LLMs). The observation is that, although LLMs are capable of answering declarative queries and extracting structured data from unstructured sources, their performance declines with complex queries, often resulting in incomplete or inaccurate outputs. Existing database optimization techniques target first cost or latency reduction without addressing adequately the challenge of ensuring result quality.

Galois implements an intermediate optimization layer between SQL and LLMs while treating the LLM as a storage layer. The system modifies both logical and physical query plans to adapt to the distinct operational characteristics of language models.

Galois presents several key innovations. First, it redefines the query optimization process to balance execution cost and result quality. It also introduces two physical operators for retrieving data from LLMs: table-scan and key-scan. Since LLMs lack catalogs, Galois dynamically builds metadata and estimates confidence scores for query execution based on the model’s self-assessed certainty.

Across multiple datasets, Galois achieved an improvement of 144% in result quality over natural language queries and of 29% over direct SQL execution, with a moderate increase in computational cost.

The main lessons we keep in mind for our implementation are:
- Introduce logical and physical plans that adapt query decomposition to the LLM's strengths and limitations
- Treat the LLM as a data source, not just a reasoning tool
- Maintain a cost-quality balance
- Structure results in JSON to ensure compatibility with analysis tools

---

## Detailed Implementation schedule

Here are the deadlines by task that we gave to ourselves according to the overall phase deadlines

**Phase 1 – Tech Selection and System Setup**

Overall deadline: 31 October 2025

| Deadline         | Task                        | Description                                                                                                                    |
| ------------- | --------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **18 Oct**    | Literature review           | Study and summarize the *Galois* SIGMOD 2025 paper. Extract core design principles and optimization strategies.                |
| **19–20 Oct** | System architecture design  | Define system components (SQL layer, LLM layer, orchestration, evaluation). Draw architecture diagram.                         |
| **21 Oct**    | Python environment setup    | Create virtual environment, install core libraries. |
| **21 Oct** | Database setup              | Install DuckDB and/or PostgreSQL, create reference databases, import provided datasets.                                        |
| **24 Oct**    | SQL connection test         | Write Python scripts to connect, query, and retrieve results in JSON format.                                                   |
| **25 Oct**    | EXPLAIN plan test           | Execute `EXPLAIN` queries and capture logical plans for analysis.                                                              |
| **30–31 Oct** | Final verification & report | Run end-to-end test and commit initial setup. Prepare README + PDF.                                                |


**Phase 2 – NL and SQL Baselines**

Overall deadline: 19 November 2025

| Date          | Task                         | Description                                                                                          |
| ------------- | ---------------------------- | ---------------------------------------------------------------------------------------------------- |
| **1–3 Nov**   | NL baseline scripting        | Write prompts for natural-language queries, ensure JSON-formatted responses.                         |
| **4–6 Nov**   | SQL baseline scripting       | Write equivalent SQL queries and test direct execution on DuckDB/PostgreSQL.                         |
| **7–8 Nov**   | LLM orchestration            | Integrate both baselines into LangChain workflow for reproducible runs.                              |
| **9–11 Nov**  | Prompt strategy testing      | Experiment with prompt variations (few-shot, chain-of-thought, direct). Evaluate parsing robustness. |
| **12–14 Nov** | Evaluation metrics setup     | Implement evaluation library to compute accuracy, recall, and F1 for NL/SQL results.                 |
| **15–17 Nov** | Unit testing                 | Build test runner to execute all queries, gather JSON outputs, and compare results.                  |
| **18–19 Nov** | Documentation & verification | Prepare README + PDF replicating first two columns of Table 3 (Galois). Upload to GitHub and Crane.  |


**Phase 3 – Implementation of Galois Features (Part I)**

Intermediate demo deadline: 5 December 2025

| Date             | Task                              | Description                                                                                                  |
| ---------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **20–22 Nov**    | Implement LLMScan Operator        | Develop class or function to query the LLM using a table name and list of attributes. Return tuples as JSON. |
| **23–26 Nov**    | Implement Filter-LLMScan Operator | Extend previous operator to handle condition pushdowns (`WHERE` clauses).                                    |
| **27–29 Nov**    | Multi-relation joining            | Add support for joins and post-processing (attribute matching, tuple merging).                               |
| **30 Nov–2 Dec** | Two-phase retrieval               | Implement key-first (Key-Scan) method based on Galois’ approach.                                             |
| **3–4 Dec**      | End-to-end system test            | Run EU Cities example (naïve vs. optimized plan) and collect metrics (tokens, accuracy, execution time).     |
| **5 Dec**        | Demo preparation                  | Create short demonstration for advisor comparing approaches and results.                                     |


**Phase 4 – Implementation of Galois Features (Part II)**

Partial deadline: 12 December 2025

Final deadline: 23 December 2025

| Date          | Task                           | Description                                                                                       |
| ------------- | ------------------------------ | ------------------------------------------------------------------------------------------------- |
| **6–8 Dec**   | Galois Without Optimizations   | Implement baseline (WO) version of Galois for replication of column 3 of Table 3.                 |
| **9–11 Dec**  | Testing and metrics            | Run experiments, collect results, and compute F1-Cell, Cardinality, and Tuple Constraint metrics. |
| **12 Dec**    | Advisor demo                   | Present informal results replicating Table 3 column 3.                                            |
| **13–16 Dec** | Implement Galois Selective     | Add logic for pushing down most selective conditions (based on model confidence).                 |
| **17–19 Dec** | Implement Galois All Attribute | Extend query optimization to all attributes and evaluate cost/quality tradeoff.                   |
| **20–22 Dec** | Implement Galois Full          | Combine logical and physical optimizations, finalize confidence-driven cost model.                |
| **23 Dec**    | Final verification and report  | Replicate complete Table 3, finalize demo and PDF report for submission.                          |

**Phase 5 – Final Presentation**

Deadline: 8 January 2026


| Date          | Task              | Description                                                                                 |
| ------------- | ----------------- | ------------------------------------------------------------------------------------------- |
| **2-4 Jan** | Slide preparation | Draft slides summarizing system architecture, implementation stages, and results.           |
| **5–7 Jan**   | Rehearsal         | Practice group presentation and finalize PDF export.                                        |
| **8 Jan**     | Final submission  | Submit PDF slides on Crane and present during group session.                      |


---

## Technology Stack

| Component | Tool / Library | Description |
|------------|----------------|--------------|
| **Language** | Python 3.11+ | Main programming language |
| **Database** | DuckDB | Lightweight, file-based SQL database optimized for analytics |
| **LLM (later)** | IBM WatsonX.ai / OpenAI API | Large Language Model integration for later stages |
| **Libraries** | `duckdb`, `pandas`, `sqlglot`, `python-dotenv` | Core Python dependencies |
| **IDE** | Visual Studio Code | Main development environment |

---

## 🛠️ Installation and Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Graph-Databases-2025-26/PathX.git
cd PathX
```

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
python src/pipelines/run_queries.py     --datasets <datasetname1> <datasetname2>

# 6. To export EXPLAIN
python src/pipelines/explain_text.py    --datasets <datasetname1> <datasetname2> # to .txt
python src/pipelines/explain_json.py    --datasets <datasetname1> <datasetname2> # to .json

# 7. Verify the evaluation (baseline == ground)
python scripts/galois_eval.py --ground ground_truth_result --submissions ground_truth_result \
  --datasets GEO MOVIES PRESIDENTS WORLD FLIGHT-2 FLIGHT-4 --cell-metric similarity --tuple-metric constraint
```