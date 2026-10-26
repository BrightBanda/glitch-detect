#!/usr/bin/env bash
# Glitch//Detect launcher — creates the venv on first run, then serves the app.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "→ creating virtual environment"
  python3 -m venv .venv
  .venv/bin/pip install --upgrade pip >/dev/null
  .venv/bin/pip install -r requirements.txt
fi

exec .venv/bin/python app.py
