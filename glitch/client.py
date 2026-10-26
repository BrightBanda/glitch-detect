"""HTTP transport for OpenAI-compatible chat-completion endpoints.

One pooled ``requests.Session`` is shared by every call (parallel chunks reuse
TLS connections). Failures are mapped to typed errors so the analyzer can tell
"this model is unavailable / rate limited" (try another) from "this key is
wrong" (stop).
"""

from __future__ import annotations

import time

import requests
from requests.adapters import HTTPAdapter

from . import config
from .errors import DetectorError, ModelFailure, RateLimited
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


def headers(api_key: str, provider: str) -> dict:
    out = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
    }
    if get_provider(provider)["key"] == "openrouter":
        out["HTTP-Referer"] = config.APP_URL
        out["X-Title"] = config.APP_TITLE
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
    kwargs = {"timeout": timeout}
    if api_key:
        kwargs["headers"] = headers(api_key, provider)
    return _session.get(url, **kwargs)


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
):
    """POST a chat completion and return the HTTP response (status 200).

    ``rate_limit_retries=0`` makes a 429 raise immediately, which lets the
    analyzer switch models instead of sleeping. ``options`` are optional
    parameters (e.g. reasoning effort); if the provider rejects the request
    with HTTP 400 they are dropped and the call is retried once without them.
    """
    cfg = get_provider(provider)
    label = cfg["label"]
    url = f"{cfg['base_url']}/chat/completions"
    base_payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
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
                headers=headers(api_key, cfg["key"]),
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
            raise DetectorError(
                f"Could not reach {label}. Check your internet connection.",
                status=502,
                detail=str(exc)[:300],
            ) from exc

        status = response.status_code
        if status == 200:
            return response

        detail = upstream_detail(response)
        if status == 429:
            retry_after = _retry_after(response)
            if rate_limited < rate_limit_retries:
                rate_limited += 1
                delay = retry_after or 3 * rate_limited
                wait(min(delay, _MAX_RETRY_AFTER_S))
                continue
            raise RateLimited(
                f"{label} rate limit reached for {model}.",
                detail=detail,
                retry_after=retry_after,
            )
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
            messages, api_key, model, cfg["key"], max_tokens=max_tokens, **kwargs
        )
        try:
            data = response.json()
        except ValueError as exc:
            raise ModelFailure(
                f"{cfg['label']} returned a non-JSON envelope.",
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
            f"{cfg['label']} returned no choices.",
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
