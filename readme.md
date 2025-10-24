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

## 🧠 Technology Stack

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