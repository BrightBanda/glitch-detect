"""Glitch//Detect launcher.

Run:  python app.py
Then open the printed URL. Application code lives in the ``glitch`` package.
"""

from __future__ import annotations

import logging
import socket
import sys

from glitch import config, create_app

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
    keys = [
        name for provider, name in config.ENV_KEY_NAMES.items() if config.env_key(provider)
    ]
    key_line = f"defaults from .env: {', '.join(keys)}" if keys else "no .env keys — add them in Settings"

    banner = f"""
\033[38;5;51m  GLITCH//DETECT\033[0m
\033[38;5;213m  LOCAL FORENSICS NODE\033[0m  ·  OpenRouter + Groq transport
\033[1;97m  ➜  {url}\033[0m
\033[38;5;245m  {key_line}
  Ctrl+C to stop.\033[0m
"""
    print(banner, flush=True)
    app.run(
        host=config.HOST,
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False,
        load_dotenv=False,  # glitch.config already read .env
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  node halted.", file=sys.stderr)
