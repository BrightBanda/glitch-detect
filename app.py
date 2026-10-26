"""Glitch//Detect — local-first AI writing forensics dashboard.

Run:  python app.py
Then open the printed URL.
"""

from __future__ import annotations

import logging
import os
import socket
import sys

from flask import Flask, jsonify, render_template, request

import detector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("glitch")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB JSON envelope
app.config["JSON_SORT_KEYS"] = False
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.jinja_env.auto_reload = True

MIN_CHARS = 80
MAX_CHARS = 60_000


@app.after_request
def harden(response):
    response.headers["Cache-Control"] = "no-store, max-age=0"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "img-src 'self' data:; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "script-src 'self'; "
        "connect-src 'self'; "
        "base-uri 'none'; "
        "form-action 'none'; "
        "frame-ancestors 'none'"
    )
    return response


@app.get("/")
def index():
    return render_template(
        "index.html",
        providers=detector.PROVIDERS,
        default_provider="openrouter",
        default_model=detector.DEFAULT_MODEL,
        max_chars=MAX_CHARS,
    )


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok": True,
            "service": "glitch-detect",
            "default_model": detector.DEFAULT_MODEL,
            "endpoint": detector.OPENROUTER_BASE_URL,
            "providers": {
                key: {
                    "label": cfg["label"],
                    "default_model": cfg["default_model"],
                    "free_only": True,
                }
                for key, cfg in detector.PROVIDERS.items()
            },
        }
    )


@app.post("/api/models")
def models():
    """Free models only, ranked for the sentence-classification task."""
    payload = request.get_json(silent=True) or {}
    provider = (payload.get("provider") or "openrouter").strip().lower()
    api_key = (payload.get("api_key") or "").strip()
    force = bool(payload.get("force"))

    try:
        data = detector.list_models(provider, api_key, force=force)
    except detector.DetectorError as exc:
        log.warning("model list failed: %s %s", exc.status, exc.message)
        return _error(exc.message, exc.status, exc.detail)
    return jsonify({"ok": True, **data})


@app.post("/api/analyze")
def analyze_endpoint():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    api_key = (payload.get("api_key") or "").strip()
    provider = (payload.get("provider") or "openrouter").strip().lower()
    model = (payload.get("model") or "").strip()
    try:
        chunk_words = int(payload.get("chunk_words") or 900)
    except (TypeError, ValueError):
        chunk_words = 900

    label = detector.PROVIDERS.get(provider, {}).get("label", "provider")
    if len(text) < MIN_CHARS:
        return _error("Paste at least a couple of sentences (80+ characters).", 400)
    if len(text) > MAX_CHARS:
        return _error(
            f"Text is too long ({len(text):,} characters). The limit is {MAX_CHARS:,}.",
            413,
        )
    if not api_key:
        return _error(f"Enter your {label} API key in the sidebar.", 400)

    try:
        result = detector.analyze(
            text=text,
            api_key=api_key,
            model=model,
            chunk_words=chunk_words,
            provider=provider,
        )
    except detector.DetectorError as exc:
        log.warning("analysis failed: %s %s", exc.status, exc.message)
        return _error(exc.message, exc.status, exc.detail)
    except Exception:
        log.exception("unexpected analysis failure")
        return _error(
            "Unexpected server error while analyzing. Check the server log.",
            500,
        )

    log.info(
        "analyzed %s words / %s sentences -> %s%% (%s / %s, %sms)",
        result["words"],
        result["total"],
        result["score"],
        result["provider"],
        result["model"],
        result["elapsed_ms"],
    )
    return jsonify({"ok": True, "result": result})


@app.errorhandler(413)
def too_large(_):
    return _error("Request too large. Trim the text and try again.", 413)


def _error(message: str, status: int, detail: str = ""):
    return (
        jsonify({"ok": False, "error": {"message": message, "detail": detail}}),
        status,
    )


def _free_port(start: int) -> int:
    for candidate in range(start, start + 25):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", candidate))
                return candidate
            except OSError:
                continue
    raise SystemExit("No free port found in range.")


def main() -> None:
    requested = int(os.environ.get("PORT", "8501"))
    port = _free_port(requested)
    url = f"http://127.0.0.1:{port}"

    banner = f"""
\033[38;5;51m
  ▄▄▄▄  ▄   ▄  ▄▄▄  ▄▄▄▄▄  ▄▄▄  ▄   ▄     ▄▄▄  ▄▄▄▄  ▄▄▄▄▄  ▄▄▄  ▄▄▄▄▄
  █  ▄▄ █   █   █   █     █   █ █   █     █  █ █  ▄▄    █   █   █   █
  █▄▄▀  █   █   █   █▄▄▄  █▄▄▄█ █▄▄▄█     █  █ █▄▄▀     █   █▄▄▄█   █
  █     █▄▄▄█ ▄▄▄▄  █     █   █    █      █▄▄█ █        █   █   █   █
\033[0m
\033[38;5;213m  LOCAL FORENSICS NODE\033[0m  ·  OpenRouter + Groq transport
\033[1;97m  ➜  {url}\033[0m
\033[38;5;245m  Ctrl+C to stop. The API key stays in your browser tab.\033[0m
"""
    print(banner, flush=True)
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True, use_reloader=False)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  node halted.", file=sys.stderr)
