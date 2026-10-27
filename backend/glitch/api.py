"""FastAPI application: routes, request models, validation, security headers.

Serves the JSON API under ``/api`` and the static frontend (``config.FRONTEND_DIR``)
from the same origin, so the local-only host check and strict CSP keep working
and no CORS is needed. Interactive API docs live at ``/docs``.

Endpoints are plain ``def`` functions: model calls use blocking HTTP, and
FastAPI runs sync endpoints in its worker thread pool.

Accounts (``users.py``) use an HttpOnly session cookie. Checks work without
an account under guest limits; rewrites need an account. Plan limits and
daily quotas are enforced here, server-side (BUSINESS_MODEL.md §6.3).
"""

from __future__ import annotations

import hmac
import logging
import re
import time
from typing import Any

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import analytics, catalogue, config, keys, users
from .analyzer import analyze
from .errors import DetectorError
from .humanize import grounding_questions, humanize
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

# Routes counted by anonymous usage analytics. Admin and bootstrap routes are
# excluded so the dashboard's own polling doesn't inflate the numbers.
_TRACKED = {
    "/": "page",
    "/api/analyze": "analyze",
    "/api/humanize": "humanize",
    "/api/models": "models",
    "/api/keys/identify": "identify",
}

TONE_LABELS = {
    "natural": "Natural",
    "casual": "Conversational",
    "academic": "Academic",
    "professional": "Professional",
    "persuasive": "Persuasive",
    "simple": "Simple English",
}


# ---------------------------------------------------------------------------
# Errors: every failure uses {"ok": false, "error": {"message", "detail"[, "code"]}}
# ---------------------------------------------------------------------------
def error_response(message: str, status: int, detail: str = "", code: str = "") -> JSONResponse:
    error = {"message": message, "detail": detail}
    if code:
        error["code"] = code
    return JSONResponse({"ok": False, "error": error}, status_code=status)


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


class SignupIn(_Body):
    email: str = ""
    password: str = ""
    name: str = ""


class LoginIn(_Body):
    email: str = ""
    password: str = ""


class ProfileIn(_Body):
    name: str | None = None


class PasswordIn(_Body):
    current: str = ""
    new: str = ""


class ConfirmIn(_Body):
    password: str = ""


class AdminUserIn(_Body):
    email: str = ""
    password: str = ""
    name: str = ""


class AdminUserPatch(_Body):
    name: str | None = None
    plan: str | None = None
    status: str | None = None
    period_end: int | None = None
    clear_period: bool = False
    suspended: bool | None = None
    notes: str | None = None


class GrantIn(_Body):
    plan: str = "premium"
    days: int = 30


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
    # Optional details from the writer, keyed by paragraph (see /api/humanize/questions).
    facts: dict[str, str] | None = None

    @field_validator("labels", mode="before")
    @classmethod
    def _labels(cls, value):
        return value if isinstance(value, list) else None

    @field_validator("facts", mode="before")
    @classmethod
    def _facts(cls, value):
        if not isinstance(value, dict):
            return None
        return {str(k)[:32]: str(v)[:600] for k, v in list(value.items())[:20] if str(v).strip()}


class QuestionsIn(_Body):
    text: str = ""
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
# Accounts, plans and analytics helpers
# ---------------------------------------------------------------------------
_UNSET = object()


def current_user(request: Request) -> dict | None:
    """The signed-in account for this request, or None for a guest."""
    user = getattr(request.state, "user", _UNSET)
    if user is _UNSET:
        token = request.cookies.get(users.SESSION_COOKIE, "")
        user = users.store.user_for_session(token) if token else None
        request.state.user = user
    return user


def require_user(request: Request) -> dict:
    user = current_user(request)
    if user is None:
        raise DetectorError("Sign in to continue.", status=401, code="auth_required")
    return user


def quota_subject(request: Request, user: dict | None) -> str:
    """Whose daily usage this request counts against."""
    if user is not None:
        return user["id"]
    visitor = getattr(request.state, "visitor", "") or request.cookies.get(analytics.COOKIE_NAME, "")
    return f"guest:{visitor if analytics.valid_client_id(visitor) else 'anonymous'}"


def track(request: Request, **fields) -> None:
    """Attach provider/words/tokens to this request's analytics record."""
    record = getattr(request.state, "analytics", None)
    if isinstance(record, dict):
        record.update({k: v for k, v in fields.items() if v is not None})


def me_payload(request: Request, user: dict | None) -> dict:
    plan = users.store.entitlements(user)
    return {
        "ok": True,
        "user": users.store.public(user) if user else None,
        "plan": plan,
        "usage": users.store.usage_today(quota_subject(request, user)),
    }


def set_session_cookie(response: JSONResponse, request: Request, token: str) -> None:
    response.set_cookie(
        users.SESSION_COOKIE,
        token,
        max_age=users.SESSION_DAYS * 86400,
        httponly=True,
        samesite="lax",
        secure=request.url.scheme == "https",
        path="/",
    )


def plan_word_limit(user: dict | None, plan: dict, words: int, action: str) -> None:
    field = "rewrite_words" if action == "rewrite" else "max_words"
    limit = plan[field]
    if words <= limit:
        return
    who = "Guests" if user is None else f"{plan['label']} accounts"
    premium = users.PLANS["premium"][field]
    upgrade = f" Premium allows up to {premium:,} words." if limit < premium else ""
    raise DetectorError(
        f"{who} can {action} up to {limit:,} words at a time; this text has "
        f"{words:,}.{upgrade} Please shorten it or work on it in parts.",
        status=413,
        code="plan_words",
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app(*, allowed_hosts: set[str] | frozenset[str] = LOCAL_HOSTS) -> FastAPI:
    app = FastAPI(
        title="Glitch//Detect API",
        version="2.0",
        description="Local-first AI writing detection and humanizing.",
    )

    # -- middleware (registered innermost first) ---------------------------
    @app.middleware("http")
    async def track_usage(request: Request, call_next):
        """Anonymous analytics: counters and a random visitor cookie, never content."""
        endpoint = _TRACKED.get(request.url.path)
        if endpoint is None or request.method != ("GET" if endpoint == "page" else "POST"):
            return await call_next(request)
        started = time.perf_counter()
        request.state.analytics = {"provider": "", "words": 0, "tokens": 0}
        visitor = request.cookies.get(analytics.COOKIE_NAME, "")
        new_visitor = False
        if not analytics.valid_client_id(visitor):
            visitor = analytics.new_client_id() if endpoint == "page" else ""
            new_visitor = bool(visitor)
        request.state.visitor = visitor

        response = await call_next(request)
        fields = request.state.analytics
        try:
            await run_in_threadpool(
                analytics.store.record,
                endpoint,
                status_code=response.status_code,
                duration_ms=round((time.perf_counter() - started) * 1000),
                client_id=visitor,
                provider=str(fields.get("provider") or ""),
                words=fields.get("words", 0),
                tokens=fields.get("tokens", 0),
            )
        except Exception:
            log.exception("could not record anonymous analytics")
        if new_visitor:
            response.set_cookie(
                analytics.COOKIE_NAME, visitor, max_age=analytics.COOKIE_MAX_AGE,
                httponly=True, samesite="lax", path="/",
            )
        return response

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
        return error_response(exc.message, exc.status, exc.detail, exc.code)

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
            "max_rewrite_words": config.MAX_REWRITE_WORDS,
            "tones": list(TONES),
            "tone_labels": TONE_LABELS,
            "voices": list(VOICES),
        }

    @app.get("/api/providers", tags=["providers"])
    def providers():
        return {"ok": True, "default": DEFAULT_PROVIDER, "providers": _providers_public()}

    @app.post("/api/keys/identify", tags=["providers"])
    def identify_key(request: Request, body: KeyIn | None = None):
        """Which provider a user's own key belongs to. The key is not stored."""
        match = keys.resolve((body or KeyIn()).key)
        track(request, provider=match.provider)
        spec = PROVIDERS[match.provider]
        return {"ok": True, "provider": spec.id, "label": spec.label, "verified": match.verified}

    @app.post("/api/models", tags=["providers"])
    def models(request: Request, body: ModelsIn | None = None):
        body = body or ModelsIn()
        provider = body.provider_id()
        track(request, provider=provider)
        data = catalogue.list_models(provider, body.request_keys()[provider], force=body.force)
        return {"ok": True, **data}

    @app.post("/api/analyze", tags=["detect"], summary="Check text for AI writing")
    def analyze_endpoint(request: Request, body: AnalyzeIn | None = None):
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
        user = current_user(request)
        plan = users.store.entitlements(user)
        plan_word_limit(user, plan, len(body.text.split()), "check")

        provider = body.provider_id()
        track(request, provider=provider)
        with users.store.reserve(quota_subject(request, user), "checks", plan["checks_per_day"]):
            result = analyze(
                body.text,
                provider=provider,
                model=body.model,
                keys=body.request_keys(),
                options=body.options(),
            )
        track(request, words=result["words"], tokens=result["usage"]["total_tokens"])
        log.info(
            "analyzed %s words / %s sentences -> %s%% (%s / %s, %s mode, %s calls,"
            " %s reused, %s tokens, %sms%s)",
            result["words"], result["total"], result["score"], result["provider"],
            result["model"], result["mode"], result["chunks"], result["reused_sentences"],
            result["usage"]["total_tokens"], result["elapsed_ms"],
            f", {len(result['fallbacks'])} fallback(s)" if result["fallbacks"] else "",
        )
        return {"ok": True, "result": result}

    @app.post("/api/humanize/questions", tags=["humanize"], summary="Details only the writer knows")
    def humanize_questions(body: QuestionsIn | None = None):
        """Optional questions per paragraph, asked before a rewrite. Zero tokens.

        Answers go back as ``facts`` on ``POST /api/humanize``, keyed by ``key``;
        they are the only new material a rewrite may add.
        """
        body = body or QuestionsIn()
        _word_limit(body.text)
        return {"ok": True, "paragraphs": grounding_questions(body.text, body.labels)}

    @app.post("/api/humanize", tags=["humanize"], summary="Rewrite flagged passages")
    def humanize_endpoint(request: Request, body: HumanizeIn | None = None):
        body = body or HumanizeIn()
        _word_limit(body.text)
        if len(body.text) < config.MIN_CHARS:
            raise DetectorError("Paste at least a couple of full sentences to rewrite.", status=400)
        user = current_user(request)
        if user is None:
            raise DetectorError(
                "Create a free account or sign in to rewrite text. Free accounts get "
                f"{users.PLANS['free']['rewrites_per_day']} rewrites a day.",
                status=401,
                code="auth_required",
            )
        plan = users.store.entitlements(user)
        tone = body.tone or "natural"
        if tone not in plan["tones"]:
            allowed = " and ".join(TONE_LABELS.get(t, t) for t in plan["tones"])
            raise DetectorError(
                f"The {TONE_LABELS.get(tone, tone)} tone is part of Premium. "
                f"Your plan includes {allowed}.",
                status=403,
                code="plan_tone",
            )
        words = len(body.text.split())
        plan_word_limit(user, plan, words, "rewrite")

        provider = body.provider_id()
        track(request, provider=provider, words=words)
        # One rewrite per request, including "try another version" (§6.3 rules 5-6).
        with users.store.reserve(user["id"], "rewrites", plan["rewrites_per_day"]):
            result = humanize(
                body.text,
                provider=provider,
                model=body.model,
                keys=body.request_keys(),
                tone=tone,
                voice=body.voice or "personal",
                only_flagged=body.only_flagged,
                labels=body.labels,
                second_pass=body.second_pass,
                fresh=body.fresh,
                options=body.options(),
                facts=body.facts,
            )
        track(request, tokens=result["usage"]["total_tokens"])
        log.info(
            "humanized %s/%s sentences in %s passage(s) (%s reused, %s tokens, score %s -> %s, %sms)",
            result["rewritten"], result["targets"], result["passages"], result["reused"],
            result["usage"]["total_tokens"], result["score_before"], result["score_after"],
            result["elapsed_ms"],
        )
        return {"ok": True, "result": result}

    # -- accounts --------------------------------------------------------
    @app.get("/api/me", tags=["account"])
    def me(request: Request):
        """The current account (or guest), its plan limits and today's usage."""
        return me_payload(request, current_user(request))

    @app.post("/api/auth/signup", tags=["account"], status_code=201)
    def signup(request: Request, body: SignupIn | None = None):
        body = body or SignupIn()
        user = users.store.create(body.email, body.password, body.name)
        response = JSONResponse(me_payload(request, user), status_code=201)
        set_session_cookie(response, request, users.store.start_session(user["id"]))
        log.info("new account %s", user["id"])
        return response

    @app.post("/api/auth/login", tags=["account"])
    def login(request: Request, body: LoginIn | None = None):
        body = body or LoginIn()
        user = users.store.authenticate(body.email, body.password)
        response = JSONResponse(me_payload(request, user))
        set_session_cookie(response, request, users.store.start_session(user["id"]))
        return response

    @app.post("/api/auth/logout", tags=["account"])
    def logout(request: Request):
        token = request.cookies.get(users.SESSION_COOKIE, "")
        if token:
            users.store.end_session(token)
        response = JSONResponse(me_payload(request, None))
        response.delete_cookie(users.SESSION_COOKIE, path="/")
        return response

    @app.patch("/api/me", tags=["account"])
    def update_me(request: Request, body: ProfileIn | None = None):
        user = require_user(request)
        user = users.store.update(user["id"], name=(body or ProfileIn()).name)
        return me_payload(request, user)

    @app.post("/api/me/password", tags=["account"])
    def change_password(request: Request, body: PasswordIn | None = None):
        user = require_user(request)
        body = body or PasswordIn()
        users.store.change_password(user["id"], body.current, body.new)
        # Sign out other devices; keep this one.
        users.store.end_all_sessions(user["id"])
        response = JSONResponse(me_payload(request, user))
        set_session_cookie(response, request, users.store.start_session(user["id"]))
        return response

    @app.post("/api/me/delete", tags=["account"])
    def delete_me(request: Request, body: ConfirmIn | None = None):
        user = require_user(request)
        users.store.authenticate(user["email"], (body or ConfirmIn()).password)
        users.store.delete(user["id"])
        log.info("account %s deleted by its owner", user["id"])
        response = JSONResponse(me_payload(request, None))
        response.delete_cookie(users.SESSION_COOKIE, path="/")
        return response

    # -- admin (see docs/ADMIN_DASHBOARD.md) --------------------------------
    admin = [Depends(require_admin)]

    def _admin_user_view(user: dict) -> dict:
        return {
            **users.store.public(user),
            "usage_today": users.store.usage_today(user["id"]),
            "limits": users.store.entitlements(user),
        }

    def _admin_get(user_id: str) -> dict:
        user = users.store.get(user_id)
        if user is None:
            raise DetectorError("No such user.", status=404)
        return user

    @app.get("/api/admin/analytics", tags=["admin"], dependencies=admin)
    def admin_analytics():
        data = analytics.store.snapshot()
        data["plans"] = users.store.summary()
        return {"ok": True, **data}

    @app.get("/api/admin/users", tags=["admin"], dependencies=admin)
    def admin_list_users(q: str = "", plan: str = "", status: str = "",
                         limit: int = 50, offset: int = 0):
        found, total = users.store.list(query=q, plan=plan, status=status, limit=limit, offset=offset)
        return {
            "ok": True,
            "users": [_admin_user_view(u) for u in found],
            "total": total,
            "plans": {k: {"label": v["label"]} for k, v in users.PLANS.items()},
            "statuses": list(users.STATUSES),
        }

    @app.post("/api/admin/users", tags=["admin"], dependencies=admin, status_code=201)
    def admin_create_user(body: AdminUserIn | None = None):
        body = body or AdminUserIn()
        user = users.store.create(body.email, body.password, body.name)
        log.info("admin created account %s", user["id"])
        return {"ok": True, "user": _admin_user_view(user)}

    @app.get("/api/admin/users/{user_id}", tags=["admin"], dependencies=admin)
    def admin_get_user(user_id: str):
        user = _admin_get(user_id)
        return {
            "ok": True,
            "user": _admin_user_view(user),
            "history": users.store.usage_history(user_id),
        }

    @app.patch("/api/admin/users/{user_id}", tags=["admin"], dependencies=admin)
    def admin_update_user(user_id: str, body: AdminUserPatch | None = None):
        body = body or AdminUserPatch()
        _admin_get(user_id)
        changes = body.model_dump(exclude={"clear_period"}, exclude_none=True)
        user = users.store.update(user_id, clear_period=body.clear_period, **changes)
        log.info("admin updated account %s: %s", user_id, sorted(changes))
        return {"ok": True, "user": _admin_user_view(user)}

    @app.post("/api/admin/users/{user_id}/grant", tags=["admin"], dependencies=admin)
    def admin_grant(user_id: str, body: GrantIn | None = None):
        body = body or GrantIn()
        user = users.store.grant(user_id, body.plan, body.days)
        log.info("admin granted %s days of %s to %s", body.days, body.plan, user_id)
        return {"ok": True, "user": _admin_user_view(user)}

    @app.post("/api/admin/users/{user_id}/reset-usage", tags=["admin"], dependencies=admin)
    def admin_reset_usage(user_id: str):
        users.store.reset_today(_admin_get(user_id)["id"])
        return {"ok": True, "user": _admin_user_view(_admin_get(user_id))}

    @app.post("/api/admin/users/{user_id}/sign-out", tags=["admin"], dependencies=admin)
    def admin_sign_out(user_id: str):
        ended = users.store.end_all_sessions(_admin_get(user_id)["id"])
        return {"ok": True, "sessions_ended": ended}

    @app.delete("/api/admin/users/{user_id}", tags=["admin"], dependencies=admin)
    def admin_delete_user(user_id: str):
        if not users.store.delete(user_id):
            raise DetectorError("No such user.", status=404)
        log.info("admin deleted account %s", user_id)
        return {"ok": True}

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
