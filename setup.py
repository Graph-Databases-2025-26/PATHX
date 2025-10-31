# full_project_setup.py
from pathlib import Path
import sys, subprocess, os

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQS_PATH = ROOT / "requirements.txt"

if os.name == "nt":
    PY  = VENV / "Scripts" / "python"
else:
    PY  = VENV / "bin" / "python"

def run(cmd: list[str | Path], *, check: bool = True) -> None:
    printable = " ".join(map(str, cmd))
    print(f"\n Running: {printable}")
    subprocess.run([str(c) for c in cmd], check=check)

def setup_environment():
    if not VENV.exists():
        print(" Creating virtual environment...")
        run([sys.executable, "-m", "venv", str(VENV)])
    else:
        print(" Virtual environment already exists")

    if REQS_PATH.exists():
        print("\n Installing dependencies from requirements.txt...")
        run([str(PY), "-m", "pip", "install", "--upgrade", "pip"], check=False)
        run([str(PY), "-m", "pip", "install", "-r", str(REQS_PATH)])
    else:
        print(f" requirements.txt not found at {REQS_PATH}; skipping installs")

def build_ground_truth():
    print("\n Building ground truth...")
    run([
        str(PY), "scripts/build_ground_truth.py",
        "--data-root", "data",
        "--ground-root", "ground_truth_result",
        "--datasets", "flight-2", "flight-4", "fortune", "geo",
        "premier", "presidents", "world", "movies"
    ])

def ingest_datasets():
    print("\n Ingesting all datasets to database...")
    run([str(PY), "src/db/ingest_all.py"])

def process_queries():
    print("\n Processing queries...")
    run([str(PY), "-m", "src/pipelines/run_queries.py"])

def export_explain():
    print("\n Exporting EXPLAIN results (.json)...")
    run([
        str(PY), "src/pipelines/explain_json.py",
        "--datasets", "flight-2", "flight-4", "fortune", "geo",
        "premier", "presidents", "world", "movies",
        "--format", "both",
        "--db-path", "./database.duckdb"
    ])

def verify_evaluation():
    print("\n Verifying evaluation (baseline == ground)...")
    run([
        str(PY), "scripts/galois_eval.py",
        "--ground", "ground",
        "--submissions", "submissions",
        "--datasets", "GEO", "MOVIES", "PRESIDENTS", "WORLD", "FLIGHT-2", "FLIGHT-4",
        "--cell-metric", "similarity",
        "--tuple-metric", "constraint",
        "--overall", "--format", "table"
    ])

def main():
    setup_environment()
    build_ground_truth()
    ingest_datasets()
    process_queries()
    export_explain()
    verify_evaluation()
    print("\n Project setup and verification complete!")

if __name__ == "__main__":
    setup_environment()
    main()
