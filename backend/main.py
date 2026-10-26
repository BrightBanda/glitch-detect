"""Glitch//Detect launcher.

Run:  python backend/main.py        (or ./run.sh from the project root)
Then open the printed URL. The API lives in the ``glitch`` package; the
frontend in ../frontend is served by the same app.

For development with auto-reload:
    uvicorn main:app --app-dir backend --reload --port 8501
"""

from __future__ import annotations

import logging
import socket
import sys

import uvicorn

from glitch import config, create_app, keys
from glitch.providers import PROVIDERS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)

app = create_app()


def _free_port(start: int) -> int:
    for candidate in range(start, start + 25):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((config.HOST, candidate))
                return candidate
            except OSError:
                continue
    raise SystemExit("No free port found in range.")


def main() -> None:
    port = _free_port(config.PORT)
    url = f"http://{config.HOST}:{port}"
    ready = sorted(PROVIDERS[p].label for p in keys.store.ready_providers())
    key_line = f"ready: {', '.join(ready)}" if ready else "no API keys yet — add one in Settings"
    if config.ADMIN_TOKEN:
        key_line += " · admin API enabled"

    banner = f"""
\033[38;5;51m  GLITCH//DETECT\033[0m
\033[38;5;213m  LOCAL FORENSICS NODE\033[0m  ·  multi-provider · FastAPI
\033[1;97m  ➜  {url}\033[0m
\033[38;5;245m  API docs: {url}/docs
  {key_line}
  Ctrl+C to stop.\033[0m
"""
    print(banner, flush=True)
    uvicorn.run(app, host=config.HOST, port=port, log_level="info")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  node halted.", file=sys.stderr)
