"""Flask application: routes, request validation, security headers."""

from __future__ import annotations

import logging

from flask import Flask, jsonify, render_template, request

from . import catalogue, config
from .analyzer import ScanOptions, analyze
from .errors import DetectorError
from .humanize import humanize
from .providers import DEFAULT_MODEL, DEFAULT_PROVIDER, PROVIDERS, get_provider

log = logging.getLogger("glitch")

# The server can spend keys from .env, so only same-machine origins may call
# it. Checking Host blocks DNS-rebinding pages from reaching the API.
_ALLOWED_HOSTS = {"127.0.0.1", "localhost"}

_CSP = (
    "default-src 'self'; "
    "img-src 'self' data:; "
    "style-src 'self' https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; "
    "script-src 'self'; "
    "connect-src 'self'; "
    "base-uri 'none'; "
    "form-action 'none'; "
    "frame-ancestors 'none'"
)


def _error(message: str, status: int, detail: str = ""):
    return (
        jsonify({"ok": False, "error": {"message": message, "detail": detail}}),
        status,
    )


def _keys_from(payload: dict, provider: str) -> dict:
    """User-supplied keys (``api_keys`` map or legacy ``api_key``) over .env."""
    supplied = payload.get("api_keys")
    supplied = supplied if isinstance(supplied, dict) else {}
    if payload.get("api_key") and not supplied.get(provider):
        supplied = {**supplied, provider: payload["api_key"]}
    return {
        p: config.resolve_key(p, str(supplied.get(p) or "")) for p in PROVIDERS
    }


def _provider_status() -> dict:
    return {
        key: {
            "label": cfg["label"],
            "default_model": cfg["default_model"],
            "keys_url": cfg["keys_url"],
            "key_placeholder": cfg["key_placeholder"],
            "env_key": bool(config.env_key(key)),
            "env_var": config.ENV_KEY_NAMES[key],
            "free_only": True,
        }
        for key, cfg in PROVIDERS.items()
    }


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(config.ROOT / "templates"),
        static_folder=str(config.ROOT / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024  # 2 MB JSON envelope
    app.json.sort_keys = False

    @app.before_request
    def local_only():
        host = (request.host or "").rsplit(":", 1)[0].strip("[]")
        if host not in _ALLOWED_HOSTS and not app.testing:
            return _error("This node only accepts local requests.", 403)

    @app.after_request
    def harden(response):
        if request.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"
        else:
            response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = _CSP
        return response

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            providers=_provider_status(),
            default_provider=DEFAULT_PROVIDER,
            default_model=DEFAULT_MODEL,
            max_chars=config.MAX_CHARS,
            min_chars=config.MIN_CHARS,
        )

    @app.get("/api/health")
    def health():
        return jsonify(
            {
                "ok": True,
                "service": "glitch-detect",
                "default_provider": DEFAULT_PROVIDER,
                "providers": _provider_status(),
            }
        )

    @app.post("/api/models")
    def models():
        """Free models only, ranked for the sentence-classification task."""
        payload = request.get_json(silent=True) or {}
        provider = str(payload.get("provider") or DEFAULT_PROVIDER).strip().lower()
        try:
            key = _keys_from(payload, get_provider(provider)["key"])[provider]
            data = catalogue.list_models(provider, key, force=bool(payload.get("force")))
        except DetectorError as exc:
            log.warning("model list failed: %s %s", exc.status, exc.message)
            return _error(exc.message, exc.status, exc.detail)
        return jsonify({"ok": True, **data})

    @app.post("/api/analyze")
    def analyze_endpoint():
        payload = request.get_json(silent=True) or {}
        text = str(payload.get("text") or "").strip()
        provider = str(payload.get("provider") or DEFAULT_PROVIDER).strip().lower()
        model = str(payload.get("model") or "").strip()

        def as_int(name: str, default: int) -> int:
            try:
                return int(payload.get(name) or default)
            except (TypeError, ValueError):
                return default

        options = ScanOptions(
            chunk_words=as_int("chunk_words", 900),
            fallback=payload.get("fallback", True) is not False,
            cross_provider=payload.get("cross_provider", True) is not False,
            concurrency=as_int("concurrency", 0),
            compact=payload.get("compact", True) is not False,
            balance=payload.get("balance", True) is not False,
        )

        if len(text) < config.MIN_CHARS:
            return _error(
                f"Paste at least a couple of sentences ({config.MIN_CHARS}+ characters).",
                400,
            )
        if len(text) > config.MAX_CHARS:
            return _error(
                f"Text is too long ({len(text):,} characters). The limit is {config.MAX_CHARS:,}.",
                413,
            )

        try:
            keys = _keys_from(payload, get_provider(provider)["key"])
            result = analyze(
                text, provider=provider, model=model, keys=keys, options=options
            )
        except DetectorError as exc:
            log.warning("analysis failed: %s %s", exc.status, exc.message)
            return _error(exc.message, exc.status, exc.detail)
        except Exception:
            log.exception("unexpected analysis failure")
            return _error(
                "Unexpected server error while analyzing. Check the server log.", 500
            )

        log.info(
            "analyzed %s words / %s sentences -> %s%% (%s / %s, %s mode, %s chunks"
            " [%s cached], %s tokens, %sms%s)",
            result["words"],
            result["total"],
            result["score"],
            result["provider"],
            result["model"],
            result["mode"],
            result["chunks"],
            result["cached_chunks"],
            result["usage"]["total_tokens"],
            result["elapsed_ms"],
            f", {len(result['fallbacks'])} fallback(s)" if result["fallbacks"] else "",
        )
        return jsonify({"ok": True, "result": result})

    @app.post("/api/humanize")
    def humanize_endpoint():
        payload = request.get_json(silent=True) or {}
        provider = str(payload.get("provider") or DEFAULT_PROVIDER).strip().lower()
        try:
            keys = _keys_from(payload, get_provider(provider)["key"])
            result = humanize(
                str(payload.get("text") or ""),
                provider=provider,
                model=str(payload.get("model") or ""),
                api_key=keys[provider],
            )
        except DetectorError as exc:
            return _error(exc.message, exc.status, exc.detail)
        return jsonify({"ok": True, "result": result})

    @app.errorhandler(413)
    def too_large(_):
        return _error("Request too large. Trim the text and try again.", 413)

    return app
