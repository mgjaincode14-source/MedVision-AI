import subprocess
import sys
import os
import venv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(PROJECT_ROOT, ".venv")

if sys.platform == "win32":
    PYTHON = os.path.join(VENV_DIR, "Scripts", "python.exe")
    PIP = os.path.join(VENV_DIR, "Scripts", "pip.exe")
else:
    PYTHON = os.path.join(VENV_DIR, "bin", "python")
    PIP = os.path.join(VENV_DIR, "bin", "pip")

def run(cmd):
    print(f"  >> {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    if result.returncode != 0:
        print(f"Error running: {' '.join(cmd)}")
        sys.exit(1)

def main():
    print("\n=== MedVision AI: Environment Setup ===\n")

    print("[1/3] Creating virtual environment...")
    if not os.path.exists(PYTHON):
        venv.create(VENV_DIR, with_pip=True, clear=True)
        print("      Created .venv")
    else:
        print("      .venv already exists, skipping.")

    print("\n[2/3] Upgrading pip...")
    run([PYTHON, "-m", "pip", "install", "--upgrade", "pip", "-q"])

    print("\n[3/3] Installing dependencies from requirements.txt...")
    req_path = os.path.join(PROJECT_ROOT, "requirements.txt")
    run([PIP, "install", "-r", req_path])

    print("\n=== Setup Complete! ===")
    print("\nTo run the preprocessing test:")
    if sys.platform == "win32":
        print("  .venv\\Scripts\\python src\\test_preprocess.py")
    else:
        print("  .venv/bin/python src/test_preprocess.py")
    print()

if __name__ == "__main__":
    main()
