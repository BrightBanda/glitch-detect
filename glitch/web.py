"""Flask application: routes, request validation, security headers."""

from __future__ import annotations

import hmac
import logging
from functools import wraps

from flask import Flask, jsonify, render_template, request

from . import catalogue, config, keys
from .analyzer import analyze
from .errors import DetectorError
from .humanize import humanize
from .prompt import TONES
from .providers import DEFAULT_MODEL, DEFAULT_PROVIDER, PROVIDERS, get_provider
from .routing import ScanOptions

log = logging.getLogger("glitch")

# The server can spend its own keys, so only same-machine origins may call
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


def _word_limit_error(text: str):
    words = len(text.split())
    if words > config.MAX_WORDS:
        return _error(
            f"That's {words:,} words. The limit is {config.MAX_WORDS:,} words per check. "
            "Please shorten it or check it in parts.",
            413,
        )
    return None


def _payload() -> dict:
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _provider_of(payload: dict) -> str:
    return get_provider(str(payload.get("provider") or DEFAULT_PROVIDER)).id


def _keys_from(payload: dict, provider: str) -> dict:
    """The user's own keys (``api_keys`` map or legacy ``api_key``) over the server's."""
    supplied = payload.get("api_keys")
    supplied = dict(supplied) if isinstance(supplied, dict) else {}
    if payload.get("api_key") and not supplied.get(provider):
        supplied[provider] = payload["api_key"]
    return keys.resolve_request_keys(supplied)


def _options(payload: dict) -> ScanOptions:
    def as_int(name: str, default: int) -> int:
        try:
            return int(payload.get(name) or default)
        except (TypeError, ValueError):
            return default

    return ScanOptions(
        chunk_words=as_int("chunk_words", 900),
        fallback=payload.get("fallback", True) is not False,
        cross_provider=payload.get("cross_provider", True) is not False,
        concurrency=as_int("concurrency", 0),
        compact=payload.get("compact", True) is not False,
        balance=payload.get("balance", True) is not False,
    )


def _providers_public() -> list[dict]:
    """Every supported provider and whether the server can use it. No keys."""
    ready = keys.store.ready_providers()
    return [
        {
            "id": spec.id,
            "label": spec.label,
            "billing": spec.billing,
            "keys_url": spec.keys_url,
            "default_model": spec.default_model,
            "ready": spec.id in ready,
        }
        for spec in PROVIDERS.values()
    ]


def _admin_only(view):
    """Admin routes exist only when ADMIN_TOKEN is set, and require it."""

    @wraps(view)
    def wrapper(*args, **kwargs):
        if not config.ADMIN_TOKEN:
            return _error("Not found.", 404)
        supplied = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
        if not hmac.compare_digest(supplied.encode(), config.ADMIN_TOKEN.encode()):
            return _error("Admin token required.", 401)
        return view(*args, **kwargs)

    return wrapper


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

    @app.errorhandler(DetectorError)
    def detector_error(exc: DetectorError):
        log.warning("%s %s: %s %s", request.method, request.path, exc.status, exc.message)
        return _error(exc.message, exc.status, exc.detail)

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            default_provider=DEFAULT_PROVIDER,
            default_model=DEFAULT_MODEL,
            max_chars=config.MAX_CHARS,
            min_chars=config.MIN_CHARS,
            max_words=config.MAX_WORDS,
            tones=TONES,
        )

    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "service": "glitch-detect", "providers": _providers_public()})

    @app.get("/api/providers")
    def providers():
        return jsonify({"ok": True, "default": DEFAULT_PROVIDER, "providers": _providers_public()})

    @app.post("/api/keys/identify")
    def identify_key():
        """Which provider a user's own key belongs to. The key is not stored."""
        match = keys.resolve(str(_payload().get("key") or ""))
        spec = PROVIDERS[match.provider]
        return jsonify(
            {"ok": True, "provider": spec.id, "label": spec.label, "verified": match.verified}
        )

    @app.post("/api/models")
    def models():
        payload = _payload()
        provider = _provider_of(payload)
        key = _keys_from(payload, provider)[provider]
        data = catalogue.list_models(provider, key, force=bool(payload.get("force")))
        return jsonify({"ok": True, **data})

    @app.post("/api/analyze")
    def analyze_endpoint():
        payload = _payload()
        text = str(payload.get("text") or "").strip()
        if len(text) < config.MIN_CHARS:
            return _error(
                f"Paste at least a couple of sentences ({config.MIN_CHARS}+ characters).", 400
            )
        if len(text) > config.MAX_CHARS:
            return _error(
                f"Text is too long ({len(text):,} characters). The limit is {config.MAX_CHARS:,}.",
                413,
            )
        if (too_long := _word_limit_error(text)) is not None:
            return too_long

        provider = _provider_of(payload)
        try:
            result = analyze(
                text,
                provider=provider,
                model=str(payload.get("model") or "").strip(),
                keys=_keys_from(payload, provider),
                options=_options(payload),
            )
        except DetectorError:
            raise
        except Exception:
            log.exception("unexpected analysis failure")
            return _error("Unexpected server error while analyzing. Check the server log.", 500)

        log.info(
            "analyzed %s words / %s sentences -> %s%% (%s / %s, %s mode, %s calls,"
            " %s reused, %s tokens, %sms%s)",
            result["words"], result["total"], result["score"], result["provider"],
            result["model"], result["mode"], result["chunks"], result["reused_sentences"],
            result["usage"]["total_tokens"], result["elapsed_ms"],
            f", {len(result['fallbacks'])} fallback(s)" if result["fallbacks"] else "",
        )
        return jsonify({"ok": True, "result": result})

    @app.post("/api/humanize")
    def humanize_endpoint():
        payload = _payload()
        text = str(payload.get("text") or "").strip()
        if (too_long := _word_limit_error(text)) is not None:
            return too_long
        if len(text) < config.MIN_CHARS:
            return _error("Paste at least a couple of full sentences to rewrite.", 400)

        provider = _provider_of(payload)
        labels = payload.get("labels")
        try:
            result = humanize(
                text,
                provider=provider,
                model=str(payload.get("model") or "").strip(),
                keys=_keys_from(payload, provider),
                tone=str(payload.get("tone") or "natural"),
                only_flagged=payload.get("only_flagged", True) is not False,
                labels=labels if isinstance(labels, list) else None,
                second_pass=bool(payload.get("second_pass")),
                options=_options(payload),
            )
        except DetectorError:
            raise
        except Exception:
            log.exception("unexpected humanize failure")
            return _error("Unexpected server error while rewriting. Check the server log.", 500)

        log.info(
            "humanized %s/%s sentences (%s reused, %s tokens, score %s -> %s, %sms)",
            result["rewritten"], result["targets"], result["reused"],
            result["usage"]["total_tokens"], result["score_before"], result["score_after"],
            result["elapsed_ms"],
        )
        return jsonify({"ok": True, "result": result})

    # -- admin: key management for a future dashboard -------------------------
    @app.get("/api/admin/keys")
    @_admin_only
    def admin_list_keys():
        return jsonify({"ok": True, "keys": keys.store.list(), "providers": _providers_public()})

    @app.post("/api/admin/keys")
    @_admin_only
    def admin_add_key():
        payload = _payload()
        entry = keys.store.add(str(payload.get("key") or ""), str(payload.get("label") or ""))
        log.info("admin added a %s key (%s)", entry["provider"], entry["key"])
        return jsonify({"ok": True, "key": entry}), 201

    @app.delete("/api/admin/keys/<entry_id>")
    @_admin_only
    def admin_remove_key(entry_id: str):
        if not keys.store.remove(entry_id):
            return _error("No such key (keys from the environment can't be removed here).", 404)
        return jsonify({"ok": True})

    @app.errorhandler(413)
    def too_large(_):
        return _error("Request too large. Trim the text and try again.", 413)

    return app
