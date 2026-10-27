"""Rate-limit awareness learned from provider responses.

Free tiers limit requests and tokens per minute *and* per day. Reacting only
after a 429 wastes requests and time (every scan re-tried dead models), so:

* **Pacing** — Groq (``x-ratelimit-remaining-tokens`` / ``-reset-tokens``) and
  OpenRouter (``X-RateLimit-Remaining`` / ``-Reset``) report what's left. The
  router skips or briefly waits for a model that can't fit the next request.
* **Daily caps** — a 429 that names a per-day limit (OpenRouter's
  ``free-models-per-day``, Groq's ``tokens per day``) blocks the model, or the
  whole provider when the cap is account-wide, until the reset time instead
  of for a minute.

State is in-memory and process-wide; it is advisory, so a restart only costs
one extra 429 per model.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass

_DURATION_RE = re.compile(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m(?!s))?(?:(\d+(?:\.\d+)?)s)?(?:(\d+(?:\.\d+)?)ms)?$")
_DAILY_RE = re.compile(r"per[- ]day|\bRPD\b|\bTPD\b|daily", re.IGNORECASE)

DEFAULT_COOLDOWN_S = 60
DAILY_FALLBACK_S = 3600          # daily cap with no reset hint: re-check hourly
MAX_COOLDOWN_S = 24 * 3600
ACCOUNT_WIDE_DAILY = {"openrouter"}  # free-model daily cap is shared by every :free model


def parse_duration(value: str) -> float | None:
    """'2m59.5s', '7.66s', '120ms', '30' → seconds."""
    value = (value or "").strip()
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    match = _DURATION_RE.match(value)
    if not match or not any(match.groups()):
        return None
    h, m, s, ms = (float(g) if g else 0.0 for g in match.groups())
    return h * 3600 + m * 60 + s + ms / 1000


def _reset_seconds(headers) -> float | None:
    """Seconds until the provider's window resets, from whichever header it sends."""
    headers = {k.lower(): v for k, v in (headers or {}).items()}
    for name in ("x-ratelimit-reset-tokens", "x-ratelimit-reset-requests"):
        seconds = parse_duration(headers.get(name, ""))
        if seconds is not None:
            return seconds
    reset = headers.get("x-ratelimit-reset", "")
    if reset:
        try:
            value = float(reset)
        except ValueError:
            return None
        now = time.time()
        if value > 1e12:      # epoch milliseconds (OpenRouter)
            return max(0.0, value / 1000 - now)
        if value > 1e9:       # epoch seconds
            return max(0.0, value - now)
        return value          # already a delay
    return None


@dataclass
class Window:
    remaining_tokens: int | None = None
    remaining_requests: int | None = None
    resets_at: float = 0.0


class Ledger:
    def __init__(self):
        self._lock = threading.Lock()
        self._windows: dict[tuple[str, str], Window] = {}
        self._blocked: dict[tuple[str, str], float] = {}  # (provider, model|"*") -> until

    # -- learning -------------------------------------------------------------
    def record(self, provider: str, model: str, headers) -> None:
        """Remember what a response says is left in the current window."""
        lowered = {k.lower(): v for k, v in (headers or {}).items()}

        def number(*names):
            for name in names:
                try:
                    return int(float(lowered[name]))
                except (KeyError, ValueError):
                    continue
            return None

        tokens = number("x-ratelimit-remaining-tokens")
        requests_left = number("x-ratelimit-remaining-requests", "x-ratelimit-remaining")
        if tokens is None and requests_left is None:
            return
        reset = _reset_seconds(headers)
        with self._lock:
            self._windows[(provider, model)] = Window(
                remaining_tokens=tokens,
                remaining_requests=requests_left,
                resets_at=time.monotonic() + (reset if reset is not None else DEFAULT_COOLDOWN_S),
            )

    def rate_limited(self, provider: str, model: str, headers, message: str,
                     retry_after: float = 0.0) -> tuple[str, float]:
        """Record a 429. Returns (scope, seconds) where scope is 'model' or 'provider'."""
        daily = bool(_DAILY_RE.search(message or ""))
        seconds = retry_after or _reset_seconds(headers) or 0.0
        if daily:
            seconds = seconds or DAILY_FALLBACK_S
            scope = "provider" if provider in ACCOUNT_WIDE_DAILY else "model"
        else:
            seconds = seconds or DEFAULT_COOLDOWN_S
            scope = "model"
        seconds = min(seconds, MAX_COOLDOWN_S)
        key = (provider, "*" if scope == "provider" else model)
        with self._lock:
            self._blocked[key] = max(self._blocked.get(key, 0.0), time.monotonic() + seconds)
            self._windows.pop((provider, model), None)
        return scope, seconds

    # -- asking ---------------------------------------------------------------
    def blocked_for(self, provider: str, model: str) -> float:
        """Seconds until this model may be tried again (0 if available)."""
        now = time.monotonic()
        with self._lock:
            until = max(self._blocked.get((provider, "*"), 0.0), self._blocked.get((provider, model), 0.0))
        return max(0.0, until - now)

    def wait_for(self, provider: str, model: str, tokens: int) -> float:
        """Seconds to wait before a request of ~``tokens`` fits this model's window."""
        blocked = self.blocked_for(provider, model)
        if blocked:
            return blocked
        now = time.monotonic()
        with self._lock:
            window = self._windows.get((provider, model))
        if window is None or window.resets_at <= now:
            return 0.0
        if window.remaining_requests is not None and window.remaining_requests <= 0:
            return window.resets_at - now
        if window.remaining_tokens is not None and window.remaining_tokens < tokens:
            return window.resets_at - now
        return 0.0

    def spend(self, provider: str, model: str, tokens: int) -> None:
        """Optimistically reserve tokens so parallel jobs don't all pick one model."""
        with self._lock:
            window = self._windows.get((provider, model))
            if window and window.remaining_tokens is not None:
                window.remaining_tokens -= tokens
            if window and window.remaining_requests is not None:
                window.remaining_requests -= 1

    def provider_blocked(self, provider: str) -> float:
        with self._lock:
            until = self._blocked.get((provider, "*"), 0.0)
        return max(0.0, until - time.monotonic())

    def clear(self) -> None:
        with self._lock:
            self._windows.clear()
            self._blocked.clear()


ledger = Ledger()
