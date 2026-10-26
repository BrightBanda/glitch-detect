#!/usr/bin/env bash
# Glitch//Detect launcher — creates the venv on first run, then serves the app
# (FastAPI backend + static frontend) on http://127.0.0.1:8501.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "→ creating virtual environment"
  python3 -m venv .venv
  .venv/bin/python -m pip install --upgrade pip >/dev/null
fi

# Install or update dependencies when requirements change.
stamp=".venv/.requirements.sha"
want="$(sha256sum backend/requirements.txt | cut -d' ' -f1)"
if [ "$(cat "$stamp" 2>/dev/null)" != "$want" ]; then
  .venv/bin/python -m pip install -q -r backend/requirements.txt
  echo "$want" > "$stamp"
fi

exec .venv/bin/python backend/main.py
