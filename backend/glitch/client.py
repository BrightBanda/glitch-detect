"""HTTP transport for OpenAI-compatible chat-completion endpoints.

One pooled ``requests.Session`` is shared by every call (parallel chunks reuse
TLS connections). Failures are mapped to typed errors so the analyzer can tell
"this model is unavailable / rate limited" (try another) from "this key is
wrong" (stop).
"""

from __future__ import annotations

import hashlib
import threading
import time

import requests
from requests.adapters import HTTPAdapter

from . import config
from .errors import DetectorError, ModelFailure, RateLimited
from .limits import ledger
from .parsing import content_to_text
from .providers import get_provider
from .text import WS_RE

_session = requests.Session()
_adapter = HTTPAdapter(pool_connections=4, pool_maxsize=16)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)

_RETRYABLE_5XX = (500, 502, 503, 520, 522, 524)
_MAX_RETRY_AFTER_S = 15
MAX_OUTPUT_TOKENS = 4096


# Which auth style worked for a key, by key fingerprint (never the key itself).
_auth_choice: dict[str, str] = {}
_auth_lock = threading.Lock()


def _key_id(api_key: str) -> str:
    return hashlib.sha256(api_key.strip().encode()).hexdigest()[:16]


def auth_styles(api_key: str, provider: str) -> list[str]:
    """The provider's auth styles, the one known to work for this key first."""
    styles = list(get_provider(provider).auth_styles)
    with _auth_lock:
        known = _auth_choice.get(_key_id(api_key))
    if known in styles:
        styles.remove(known)
        styles.insert(0, known)
    return styles


def _remember_auth(api_key: str, style: str) -> None:
    with _auth_lock:
        _auth_choice[_key_id(api_key)] = style


def headers(api_key: str, provider: str, style: str | None = None) -> dict:
    style = style or auth_styles(api_key, provider)[0]
    out = {"Content-Type": "application/json"}
    if style == "google":
        out["x-goog-api-key"] = api_key.strip()
    else:
        out["Authorization"] = f"Bearer {api_key.strip()}"
    out.update(get_provider(provider).extra_headers)
    return out


def upstream_detail(response) -> str:
    try:
        data = response.json()
    except Exception:
        return WS_RE.sub(" ", response.text or "")[:400]
    if isinstance(data, dict):
        error = data.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("code") or ""
            return WS_RE.sub(" ", str(message))[:400]
        if isinstance(error, str):
            return WS_RE.sub(" ", error)[:400]
    return WS_RE.sub(" ", response.text or "")[:400]


def _retry_after(response) -> float:
    value = (response.headers or {}).get("Retry-After", "")
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def get(url: str, *, api_key: str = "", provider: str = "", timeout: float = 10):
    """GET with the provider's auth, trying its other auth style after a 401/403."""
    if not api_key:
        return _session.get(url, timeout=timeout)
    response = None
    for style in auth_styles(api_key, provider):
        response = _session.get(url, headers=headers(api_key, provider, style), timeout=timeout)
        if response.status_code not in (401, 403):
            if response.status_code == 200:
                _remember_auth(api_key, style)
            return response
    return response


def post_chat(
    messages: list[dict],
    api_key: str,
    model: str,
    provider: str = "openrouter",
    *,
    rate_limit_retries: int = 2,
    deadline: float | None = None,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    options: dict | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
):
    """POST a chat completion and return the HTTP response (status 200).

    ``rate_limit_retries=0`` makes a 429 raise immediately, which lets the
    analyzer switch models instead of sleeping. ``options`` are optional
    parameters (e.g. reasoning effort); if the provider rejects the request
    with HTTP 400 they are dropped and the call is retried once without them.
    """
    cfg = get_provider(provider)
    label = cfg.label
    url = f"{cfg.base_url}/chat/completions"
    base_payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max(16, min(int(max_tokens), MAX_OUTPUT_TOKENS)),
    }
    payload = {**base_payload, **(options or {})}

    def remaining() -> float:
        return float("inf") if deadline is None else deadline - time.monotonic()

    def wait(seconds: float) -> None:
        if remaining() <= seconds + 5:
            raise DetectorError(
                "The scan ran out of time while waiting on the provider.", status=504
            )
        time.sleep(seconds)

    rate_limited = 0
    server_errors = 0
    timeouts = 0
    styles = auth_styles(api_key, cfg.id)
    while True:
        if remaining() <= 5:
            raise DetectorError(
                "The scan ran out of time. Try a shorter passage or another model.",
                status=504,
            )
        read_timeout = min(config.CHUNK_TIMEOUT_S, max(5.0, remaining() - 2))
        try:
            response = _session.post(
                url,
                headers=headers(api_key, cfg.id, styles[0]),
                json=payload,
                timeout=(10, read_timeout),
            )
        except requests.Timeout as exc:
            timeouts += 1
            if timeouts > 1:
                raise ModelFailure(
                    f"{label} timed out on {model}. Free models can be slow.",
                    status=504,
                ) from exc
            continue
        except requests.RequestException as exc:
            # One unreachable provider shouldn't end the scan when another is
            # available, so this is a model-level failure the router can skip.
            raise ModelFailure(
                f"Could not reach {label}. Check your internet connection.",
                status=502,
                detail=str(exc)[:300],
            ) from exc

        status = response.status_code
        ledger.record(cfg.id, model, response.headers)
        if status == 200:
            _remember_auth(api_key, styles[0])
            return response

        detail = upstream_detail(response)
        if status == 429:
            scope, seconds = ledger.rate_limited(
                cfg.id, model, response.headers, detail, _retry_after(response)
            )
            # Wait in place only for short per-minute windows; daily caps and
            # long resets go back to the router, which picks another model.
            if rate_limited < rate_limit_retries and scope == "model" and seconds <= _MAX_RETRY_AFTER_S:
                rate_limited += 1
                wait(max(1.0, seconds))
                continue
            raise RateLimited(
                f"{label} rate limit reached for {model}.",
                detail=detail,
                retry_after=seconds,
                scope=scope,
            )
        if status in (401, 403) and len(styles) > 1:
            styles = styles[1:]  # try the provider's other way of sending the key
            continue
        if status in (401, 403):
            raise DetectorError(
                f"{label} rejected the API key. Check that the key is valid and has access to this model.",
                status=401,
                detail=detail,
            )
        if status == 402:
            raise DetectorError(
                f"{label} refused the request (HTTP 402). Check the account's plan or credits.",
                status=402,
                detail=detail,
            )
        if status == 400 and payload != base_payload:
            payload = base_payload
            continue
        if status in (400, 422):
            # The request shape suits some models and not others.
            raise ModelFailure(
                f"{label} rejected the request for {model} (HTTP {status}).",
                status=502,
                detail=detail,
            )
        if status == 404:
            raise ModelFailure(
                f"{model} is not available on {label} (it may no longer be free).",
                status=404,
                detail=detail,
            )
        if status in _RETRYABLE_5XX:
            server_errors += 1
            if server_errors <= 1:
                wait(2)
                continue
            raise ModelFailure(
                f"{label} returned HTTP {status} for {model}.", status=502, detail=detail
            )

        raise DetectorError(
            f"{label} returned HTTP {status}.",
            status=502,
            detail=detail,
        )


def _add_usage(usage: dict | None, data: dict) -> None:
    if usage is None or not isinstance(data.get("usage"), dict):
        return
    for field in ("prompt_tokens", "completion_tokens"):
        value = data["usage"].get(field)
        if isinstance(value, int):
            usage[field] = usage.get(field, 0) + value


def request_chunk(
    messages: list[dict],
    api_key: str,
    model: str,
    provider: str = "openrouter",
    *,
    usage: dict | None = None,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    **kwargs,
) -> str:
    """Send one chunk and return the assistant message text.

    Token counts reported by the provider are added to ``usage``. A reply cut
    off by ``max_tokens`` is retried once with a larger budget.
    """
    cfg = get_provider(provider)
    while True:
        response = post_chat(
            messages, api_key, model, cfg.id, max_tokens=max_tokens, **kwargs
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise ModelFailure(
                f"{cfg.label} returned a non-JSON envelope.",
                status=502,
                detail=str(exc),
            ) from exc
        if isinstance(data, dict):
            _add_usage(usage, data)
        choices = data.get("choices") if isinstance(data, dict) else None
        truncated = bool(choices) and choices[0].get("finish_reason") == "length"
        if truncated and max_tokens < MAX_OUTPUT_TOKENS:
            max_tokens = min(MAX_OUTPUT_TOKENS, max_tokens * 2 + 512)
            continue
        break

    if isinstance(data, dict) and data.get("error") and not data.get("choices"):
        raise ModelFailure(
            "The model provider returned an error.",
            status=502,
            detail=WS_RE.sub(" ", str(data.get("error")))[:400],
        )

    if not choices:
        raise ModelFailure(
            f"{cfg.label} returned no choices.",
            status=502,
            detail=str(data)[:300],
        )

    message = choices[0].get("message") or {}
    content = content_to_text(message.get("content"))
    if not content.strip():
        if message.get("reasoning"):
            raise ModelFailure(
                "The model spent its budget on reasoning and returned no JSON.",
                status=502,
            )
        raise ModelFailure("The model returned an empty message.", status=502)
    return content
