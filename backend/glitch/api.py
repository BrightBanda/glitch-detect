"""FastAPI application: routes, request models, validation, security headers.

Serves the JSON API under ``/api`` and the static frontend (``config.FRONTEND_DIR``)
from the same origin, so the local-only host check and strict CSP keep working
and no CORS is needed. Interactive API docs live at ``/docs``.

Endpoints are plain ``def`` functions: model calls use blocking HTTP, and
FastAPI runs sync endpoints in its worker thread pool.
"""

from __future__ import annotations

import hmac
import logging
import re
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, field_validator
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import catalogue, config, keys
from .analyzer import analyze
from .errors import DetectorError
from .humanize import humanize
from .prompt import TONES, VOICES
from .providers import DEFAULT_MODEL, DEFAULT_PROVIDER, PROVIDERS, get_provider
from .routing import ScanOptions

log = logging.getLogger("glitch")

MAX_BODY_BYTES = 2 * 1024 * 1024  # 2 MB JSON envelope

# The server can spend its own keys, so only same-machine hosts may call it.
# Checking Host blocks DNS-rebinding pages from reaching the API.
LOCAL_HOSTS = frozenset({"127.0.0.1", "localhost"})

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
# Swagger UI / ReDoc load their bundles from a CDN with an inline bootstrap.
_DOCS_CSP = (
    "default-src 'self'; "
    "img-src 'self' data: https://fastapi.tiangolo.com; "
    "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net https://fonts.googleapis.com; "
    "font-src https://fonts.gstatic.com; "
    "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
    "connect-src 'self'; "
    "frame-ancestors 'none'"
)
_DOCS_PATHS = ("/docs", "/redoc", "/openapi.json")
_PAGE_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")


# ---------------------------------------------------------------------------
# Errors: every failure uses {"ok": false, "error": {"message", "detail"}}
# ---------------------------------------------------------------------------
def error_response(message: str, status: int, detail: str = "") -> JSONResponse:
    return JSONResponse(
        {"ok": False, "error": {"message": message, "detail": detail}}, status_code=status
    )


def _word_limit(text: str) -> None:
    words = len(text.split())
    if words > config.MAX_WORDS:
        raise DetectorError(
            f"That's {words:,} words. The limit is {config.MAX_WORDS:,} words per check. "
            "Please shorten it or check it in parts.",
            status=413,
        )


# ---------------------------------------------------------------------------
# Request models — lenient like the original API: unknown fields are ignored
# and malformed numbers fall back to defaults.
# ---------------------------------------------------------------------------
class _Body(BaseModel):
    model_config = ConfigDict(extra="ignore")


class KeyIn(_Body):
    key: str = ""
    label: str = ""


class _ProviderBody(_Body):
    provider: str = DEFAULT_PROVIDER
    api_keys: dict[str, Any] = {}
    api_key: str = ""  # legacy single-key field

    @field_validator("provider", mode="before")
    @classmethod
    def _provider_id(cls, value):
        return str(value or DEFAULT_PROVIDER).strip().lower()

    @field_validator("api_keys", mode="before")
    @classmethod
    def _keys_map(cls, value):
        return value if isinstance(value, dict) else {}

    def provider_id(self) -> str:
        return get_provider(self.provider).id

    def request_keys(self) -> dict[str, str]:
        """The user's own keys over the server's, per provider."""
        supplied = {p: str(k or "") for p, k in self.api_keys.items()}
        provider = self.provider_id()
        if self.api_key and not supplied.get(provider):
            supplied[provider] = self.api_key
        return keys.resolve_request_keys(supplied)


class ModelsIn(_ProviderBody):
    force: bool = False


class _RoutedBody(_ProviderBody):
    text: str = ""
    model: str = ""
    chunk_words: int = 900
    concurrency: int = 0
    fallback: bool = True
    cross_provider: bool = True
    compact: bool = True
    balance: bool = True

    @field_validator("chunk_words", "concurrency", mode="before")
    @classmethod
    def _lenient_int(cls, value, info):
        try:
            return int(value)
        except (TypeError, ValueError):
            return 900 if info.field_name == "chunk_words" else 0

    @field_validator("text", "model", mode="before")
    @classmethod
    def _stripped(cls, value):
        return str(value or "").strip()

    def options(self) -> ScanOptions:
        return ScanOptions(
            chunk_words=self.chunk_words or 900,
            fallback=self.fallback,
            cross_provider=self.cross_provider,
            concurrency=self.concurrency,
            compact=self.compact,
            balance=self.balance,
        )


class AnalyzeIn(_RoutedBody):
    pass


class HumanizeIn(_RoutedBody):
    tone: str = "natural"
    voice: str = "personal"
    only_flagged: bool = True
    second_pass: bool = True
    fresh: bool = False
    labels: list[dict[str, Any]] | None = None

    @field_validator("labels", mode="before")
    @classmethod
    def _labels(cls, value):
        return value if isinstance(value, list) else None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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


def require_admin(request: Request) -> None:
    """Admin routes exist only when ADMIN_TOKEN is set, and require it."""
    if not config.ADMIN_TOKEN:
        raise DetectorError("Not found.", status=404)
    supplied = request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
    if not hmac.compare_digest(supplied.encode(), config.ADMIN_TOKEN.encode()):
        raise DetectorError("Admin token required.", status=401)


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app(*, allowed_hosts: set[str] | frozenset[str] = LOCAL_HOSTS) -> FastAPI:
    app = FastAPI(
        title="Glitch//Detect API",
        version="2.0",
        description="Local-first AI writing detection and humanizing.",
    )

    # -- middleware (outermost runs first) --------------------------------
    @app.middleware("http")
    async def guard(request: Request, call_next):
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]")
        if host not in allowed_hosts:
            return error_response("This node only accepts local requests.", 403)
        if int(request.headers.get("content-length") or 0) > MAX_BODY_BYTES:
            return error_response("Request too large. Trim the text and try again.", 413)
        # Writes must be JSON: a cross-site <form> or text/plain POST can't
        # spend the server's keys.
        if request.method in ("POST", "PUT", "PATCH") and request.url.path.startswith("/api/"):
            content_type = request.headers.get("content-type", "")
            if not content_type.split(";")[0].strip().endswith("json"):
                return error_response("Requests must be sent as JSON.", 415)
        return await call_next(request)

    @app.middleware("http")
    async def harden(request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        response.headers["Cache-Control"] = (
            "no-cache" if path.startswith("/static/") else "no-store, max-age=0"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            _DOCS_CSP if path.startswith(_DOCS_PATHS) else _CSP
        )
        return response

    # -- error handlers --------------------------------------------------
    @app.exception_handler(DetectorError)
    async def detector_error(request: Request, exc: DetectorError):
        log.warning("%s %s: %s %s", request.method, request.url.path, exc.status, exc.message)
        return error_response(exc.message, exc.status, exc.detail)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError):
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e.get('loc', ())[1:]) or 'body'}: {e.get('msg', '')}"
            for e in exc.errors()
        )
        return error_response("The request wasn't understood.", 400, problems[:400])

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException):
        messages = {404: "Not found.", 405: "Method not allowed."}
        return error_response(messages.get(exc.status_code, str(exc.detail)), exc.status_code)

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception):
        log.exception("unexpected failure on %s %s", request.method, request.url.path)
        return error_response("Unexpected server error. Check the server log.", 500)

    # -- public API ------------------------------------------------------
    @app.get("/api/health", tags=["status"])
    def health():
        return {"ok": True, "service": "glitch-detect", "providers": _providers_public()}

    @app.get("/api/config", tags=["status"])
    def client_config():
        """Limits and defaults the frontend needs at start-up."""
        return {
            "ok": True,
            "default_provider": DEFAULT_PROVIDER,
            "default_model": DEFAULT_MODEL,
            "min_chars": config.MIN_CHARS,
            "max_chars": config.MAX_CHARS,
            "max_words": config.MAX_WORDS,
            "tones": list(TONES),
            "voices": list(VOICES),
        }

    @app.get("/api/providers", tags=["providers"])
    def providers():
        return {"ok": True, "default": DEFAULT_PROVIDER, "providers": _providers_public()}

    @app.post("/api/keys/identify", tags=["providers"])
    def identify_key(body: KeyIn | None = None):
        """Which provider a user's own key belongs to. The key is not stored."""
        match = keys.resolve((body or KeyIn()).key)
        spec = PROVIDERS[match.provider]
        return {"ok": True, "provider": spec.id, "label": spec.label, "verified": match.verified}

    @app.post("/api/models", tags=["providers"])
    def models(body: ModelsIn | None = None):
        body = body or ModelsIn()
        provider = body.provider_id()
        data = catalogue.list_models(provider, body.request_keys()[provider], force=body.force)
        return {"ok": True, **data}

    @app.post("/api/analyze", tags=["detect"], summary="Check text for AI writing")
    def analyze_endpoint(body: AnalyzeIn | None = None):
        body = body or AnalyzeIn()
        if len(body.text) < config.MIN_CHARS:
            raise DetectorError(
                f"Paste at least a couple of sentences ({config.MIN_CHARS}+ characters).",
                status=400,
            )
        if len(body.text) > config.MAX_CHARS:
            raise DetectorError(
                f"Text is too long ({len(body.text):,} characters). The limit is {config.MAX_CHARS:,}.",
                status=413,
            )
        _word_limit(body.text)

        provider = body.provider_id()
        result = analyze(
            body.text,
            provider=provider,
            model=body.model,
            keys=body.request_keys(),
            options=body.options(),
        )
        log.info(
            "analyzed %s words / %s sentences -> %s%% (%s / %s, %s mode, %s calls,"
            " %s reused, %s tokens, %sms%s)",
            result["words"], result["total"], result["score"], result["provider"],
            result["model"], result["mode"], result["chunks"], result["reused_sentences"],
            result["usage"]["total_tokens"], result["elapsed_ms"],
            f", {len(result['fallbacks'])} fallback(s)" if result["fallbacks"] else "",
        )
        return {"ok": True, "result": result}

    @app.post("/api/humanize", tags=["humanize"], summary="Rewrite flagged passages")
    def humanize_endpoint(body: HumanizeIn | None = None):
        body = body or HumanizeIn()
        _word_limit(body.text)
        if len(body.text) < config.MIN_CHARS:
            raise DetectorError("Paste at least a couple of full sentences to rewrite.", status=400)

        provider = body.provider_id()
        result = humanize(
            body.text,
            provider=provider,
            model=body.model,
            keys=body.request_keys(),
            tone=body.tone or "natural",
            voice=body.voice or "personal",
            only_flagged=body.only_flagged,
            labels=body.labels,
            second_pass=body.second_pass,
            fresh=body.fresh,
            options=body.options(),
        )
        log.info(
            "humanized %s/%s sentences in %s passage(s) (%s reused, %s tokens, score %s -> %s, %sms)",
            result["rewritten"], result["targets"], result["passages"], result["reused"],
            result["usage"]["total_tokens"], result["score_before"], result["score_after"],
            result["elapsed_ms"],
        )
        return {"ok": True, "result": result}

    # -- admin: key management (see docs/ADMIN_DASHBOARD.md) ----------------
    admin = [Depends(require_admin)]

    @app.get("/api/admin/keys", tags=["admin"], dependencies=admin)
    def admin_list_keys():
        return {"ok": True, "keys": keys.store.list(), "providers": _providers_public()}

    @app.post("/api/admin/keys", tags=["admin"], dependencies=admin, status_code=201)
    def admin_add_key(body: KeyIn | None = None):
        body = body or KeyIn()
        entry = keys.store.add(body.key, body.label)
        log.info("admin added a %s key (%s)", entry["provider"], entry["key"])
        return {"ok": True, "key": entry}

    @app.delete("/api/admin/keys/{entry_id}", tags=["admin"], dependencies=admin)
    def admin_remove_key(entry_id: str):
        if not keys.store.remove(entry_id):
            raise DetectorError(
                "No such key (keys from the environment can't be removed here).", status=404
            )
        return {"ok": True}

    # -- frontend --------------------------------------------------------
    frontend = config.FRONTEND_DIR
    if (frontend / "static").is_dir():
        app.mount("/static", StaticFiles(directory=frontend / "static"), name="static")

    @app.get("/", include_in_schema=False)
    def index():
        return _page("index")

    @app.get("/{page}", include_in_schema=False)
    def page(page: str):
        """Any ``frontend/<page>.html`` is served at ``/<page>`` (e.g. /admin)."""
        if not _PAGE_RE.match(page):
            raise DetectorError("Not found.", status=404)
        return _page(page)

    def _page(name: str) -> FileResponse:
        path = frontend / f"{name}.html"
        if not path.is_file():
            raise DetectorError("Not found.", status=404)
        return FileResponse(path, media_type="text/html")

    return app
