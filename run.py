#!/usr/bin/env python3
"""
run.py — Cross-platform launcher for assignment2 scripts.
Works on Linux, macOS, and Windows, in any shell.
Handles PEP 668 (externally managed Python) by always using a venv.

First time setup (any OS):
    python run.py --setup

Then run tasks:
    python run.py task1_data.py
    python run.py task2_signal.py
    python run.py task3_model.py
    python run.py task4_analysis.py
"""

import sys
import os
import subprocess
import venv

VENV_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")

# venv python binary path differs between Linux/macOS and Windows
if sys.platform == "win32":
    VENV_PYTHON = os.path.join(VENV_DIR, "Scripts", "python.exe")
else:
    VENV_PYTHON = os.path.join(VENV_DIR, "bin", "python")

def setup():
    # Create venv if it doesn't exist
    if not os.path.exists(VENV_PYTHON):
        print("Creating virtual environment...")
        venv.create(VENV_DIR, with_pip=True)
        print("Virtual environment created.")
    else:
        print("Virtual environment already exists, skipping creation.")

    # Install packages into the venv — bypasses PEP 668 entirely
    # because we're using the venv's own pip, not the system pip
    print("Installing required packages into venv...")
    subprocess.run(
        [VENV_PYTHON, "-m", "pip", "install", "-r", "requirements.txt"],
        check=True
    )
    print("\nSetup complete! You can now run tasks with:")
    print("    python run.py task1_data.py")

def run_script(script):
    if not os.path.exists(VENV_PYTHON):
        print("Venv not found. Run setup first:")
        print("    python run.py --setup")
        sys.exit(1)

    if not os.path.exists(script):
        print(f"Error: '{script}' not found in current directory.")
        sys.exit(1)

    result = subprocess.run([VENV_PYTHON, script] + sys.argv[2:])
    sys.exit(result.returncode)

def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        sys.exit(0)

    if sys.argv[1] == "--setup":
        setup()
    else:
        run_script(sys.argv[1])

if __name__ == "__main__":
    main()