"""Runtime configuration: ``.env`` loading, server limits, storage paths.

Provider endpoints and key variable names live in ``providers.py``; key
handling lives in ``keys.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Minimal ``KEY=VALUE`` reader. Real environment variables take precedence."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, value = (part.strip() for part in line.split("=", 1))
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].strip()
        if key:
            os.environ.setdefault(key, value)


load_dotenv()

HOST = "127.0.0.1"
PORT = int(os.environ.get("PORT", "8501"))
APP_TITLE = "Glitch//Detect"
APP_URL = "http://localhost:8501"

# Server-side state (admin-added API keys). Kept out of version control.
INSTANCE_DIR = Path(os.environ.get("GLITCH_INSTANCE_DIR", ROOT / "instance"))

# Enables the /api/admin/* endpoints when set; requests must send
# "Authorization: Bearer <ADMIN_TOKEN>".
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "").strip()

MIN_CHARS = 80
MAX_CHARS = 60_000
# Per check / rewrite. Keeps a single request inside free-tier limits.
MAX_WORDS = 2500
MIN_KEY_LENGTH = 12

# Whole-scan budget. The browser aborts at 240 s, so the server stops first
# instead of burning free-tier quota on a result nobody will read.
SCAN_DEADLINE_S = 225
CHUNK_TIMEOUT_S = 150
