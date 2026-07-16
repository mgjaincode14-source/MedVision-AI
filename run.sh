#!/bin/bash
# MedVision AI launcher — always runs inside the project venv
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="$SCRIPT_DIR/.venv/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "[setup] Virtual environment not found. Creating it now..."
    python3 -m venv "$SCRIPT_DIR/.venv"
    "$SCRIPT_DIR/.venv/bin/pip" install --upgrade pip -q
    "$SCRIPT_DIR/.venv/bin/pip" install -r "$SCRIPT_DIR/requirements.txt" -q
    echo "[setup] Done."
fi

cd "$SCRIPT_DIR"
"$VENV_PYTHON" "$@"
