from pathlib import Path
import sys, subprocess, os

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
REQS_PATH = ROOT / "requirements.txt"

if os.name == "nt":
    PIP = VENV / "Scripts" / "pip"
    PY  = VENV / "Scripts" / "python"
else:
    PIP = VENV / "bin" / "pip"
    PY  = VENV / "bin" / "python"

def run(cmd: list[str | Path], *, check: bool = True) -> None:
    printable = " ".join(map(str, cmd))
    print(f"\\nRunning: {printable}")
    subprocess.run([str(c) for c in cmd], check=check)

def setup_environment() -> None:
    if not VENV.exists():
        print("Creating virtual environment...")
        run([sys.executable, "-m", "venv", str(VENV)])
    else:
        print("Virtual environment already exists")

    if REQS_PATH.exists():
        print("\\nInstalling dependencies from requirements.txt...")
        run([str(PY), "-m", "pip", "install", "--upgrade", "pip"], check=False)
        run([str(PY), "-m", "pip", "install", "-r", str(REQS_PATH)])
    else:
        print(f"requirements.txt not found at {REQS_PATH}; skipping installs")

    print("\\nSetup complete!\\n")

def main():
    print("Hello! This is your main program running inside the virtual environment.")
    print("Program finished successfully.")

if __name__ == "__main__":
    setup_environment()
    main()
