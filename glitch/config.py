"""Runtime configuration: ``.env`` loading, provider endpoints, server limits.

Keys found in the environment (or ``.env``) are server-side defaults. They are
never sent to the browser; the UI only learns *whether* a default exists.
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

OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
).rstrip("/")
GROQ_BASE_URL = os.environ.get(
    "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
).rstrip("/")

ENV_KEY_NAMES = {"openrouter": "OPENROUTER_API_KEY", "groq": "GROQ_API_KEY"}
# Also accepted, checked after the canonical name.
ENV_KEY_ALIASES = {
    "openrouter": ("OPEN_ROUTER_KEY", "OPENROUTER_KEY", "OPEN_ROUTER_API_KEY"),
    "groq": ("GROQ_KEY",),
}

MIN_CHARS = 80
MAX_CHARS = 60_000
MIN_KEY_LENGTH = 12

# Whole-scan budget. The browser aborts at 240 s, so the server stops first
# instead of burning free-tier quota on a result nobody will read.
SCAN_DEADLINE_S = 225
CHUNK_TIMEOUT_S = 150


def env_key(provider: str) -> str:
    for name in (ENV_KEY_NAMES.get(provider, ""), *ENV_KEY_ALIASES.get(provider, ())):
        value = os.environ.get(name, "").strip() if name else ""
        if value:
            return value
    return ""


def resolve_key(provider: str, user_key: str = "") -> str:
    """The user's key for this provider, else the ``.env`` default."""
    user_key = (user_key or "").strip()
    return user_key or env_key(provider)
