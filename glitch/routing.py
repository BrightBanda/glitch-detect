"""Task-agnostic model routing shared by detection and humanize.

A feature describes its work as ``Job``s (messages + how to parse the reply +
how many output tokens it needs). ``run_jobs`` executes them in parallel and
handles everything provider-related:

* **Load balancing** — jobs are spread round-robin over the selected
  provider's top healthy models, multiplying per-model rate-limit budgets.
* **Fallback** — a model that is rate limited, gone, or failing is retired for
  the rest of the run; jobs move on immediately instead of sleeping on
  ``Retry-After``. Rate-limited models also enter a process-wide cooldown.
  Fallback never moves work onto a *paid* provider the user didn't select.
* **Token budgets** — each job requests only the output it needs, and
  reasoning models are asked for low effort where supported.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

from . import catalogue, client, config
from .errors import DetectorError, ModelFailure, ParseError, RateLimited
from .providers import PAID, PROVIDERS

DEFAULT_COOLDOWN_S = 60
MAX_CONCURRENCY = 4
CROSS_PROVIDER_MODELS = 2

_cooldowns: dict[tuple[str, str], float] = {}
_cooldown_lock = threading.Lock()


def cooldown_remaining(provider: str, model: str) -> float:
    with _cooldown_lock:
        return max(0.0, _cooldowns.get((provider, model), 0.0) - time.monotonic())


def start_cooldown(provider: str, model: str, seconds: float) -> None:
    with _cooldown_lock:
        _cooldowns[(provider, model)] = time.monotonic() + (seconds or DEFAULT_COOLDOWN_S)


def clear_cooldowns() -> None:
    with _cooldown_lock:
        _cooldowns.clear()


@dataclass(frozen=True)
class Candidate:
    provider: str
    model: str

    def __str__(self) -> str:
        return f"{PROVIDERS[self.provider].label} · {self.model}"


@dataclass
class ScanOptions:
    chunk_words: int = 900
    fallback: bool = True
    cross_provider: bool = True
    concurrency: int = 0  # 0 = provider default
    compact: bool = True
    balance: bool = True


@dataclass
class Job:
    """One request's worth of work for any feature."""

    messages: list[dict]
    parse: Callable[[str], tuple[Any, list[str]]]  # raw reply -> (value, warnings)
    budget: Callable[["Candidate"], int]           # max_tokens for a candidate
    temperature: float = 0.0


@dataclass
class JobResult:
    value: Any
    warnings: list[str]
    candidate: Candidate
    usage: dict = field(default_factory=dict)


class ModelRouter:
    """Thread-safe model choice shared by every job of one run."""

    def __init__(
        self, candidates: list[Candidate], *, balance_width: int = 1, fallback: bool = True
    ):
        self._candidates = candidates
        self.fallback = fallback
        self._primary = candidates[0].provider
        self._balance_width = max(1, balance_width)
        self._retired: set[Candidate] = set()
        self._lock = threading.Lock()
        self.switches: list[dict] = []

    def _usable(self, tried: set[Candidate]) -> list[Candidate]:
        return [c for c in self._candidates if c not in self._retired and c not in tried]

    def pick(self, tried: set[Candidate], slot: int = 0) -> Candidate | None:
        """Best candidate for a job; ``slot`` spreads jobs across models."""
        with self._lock:
            usable = self._usable(tried)
        if not usable:
            return None
        fresh = [c for c in usable if not cooldown_remaining(c.provider, c.model)]
        pool = fresh or usable
        if self._balance_width > 1:
            # Balance only within the selected provider; others stay fallbacks.
            spread = [c for c in pool if c.provider == self._primary][: self._balance_width]
            if spread:
                return spread[slot % len(spread)]
        return pool[0]

    def has_alternative(self, tried: set[Candidate]) -> bool:
        with self._lock:
            return len(self._usable(tried)) > 0

    def retire(self, candidate: Candidate, exc: DetectorError) -> None:
        if isinstance(exc, RateLimited):
            start_cooldown(candidate.provider, candidate.model, exc.retry_after)
        with self._lock:
            self._retired.add(candidate)

    def record_switch(self, src: Candidate, dst: Candidate, reason: str) -> None:
        with self._lock:
            if not any(s["from"] == str(src) and s["to"] == str(dst) for s in self.switches):
                self.switches.append({"from": str(src), "to": str(dst), "reason": reason})


def build_candidates(
    provider: str, model: str, keys: dict, options: ScanOptions, *, jobs: int = 1
) -> list[Candidate]:
    out = [Candidate(provider, model)]
    if not (options.fallback or (options.balance and jobs > 1)):
        return out

    def add(c: Candidate) -> None:
        if c not in out:
            out.append(c)

    for alternative in catalogue.fallback_models(provider, keys.get(provider, "")):
        add(Candidate(provider, alternative))
    if options.fallback and options.cross_provider:
        for other, spec in PROVIDERS.items():
            key = keys.get(other, "")
            # Never spend a paid account the user didn't choose.
            if other == provider or spec.billing == PAID or len(key) < config.MIN_KEY_LENGTH:
                continue
            for alternative in catalogue.fallback_models(other, key)[:CROSS_PROVIDER_MODELS]:
                add(Candidate(other, alternative))
    return out


def make_router(
    provider: str, model: str, keys: dict, options: ScanOptions, *, jobs: int
) -> ModelRouter:
    if jobs == 0:
        return ModelRouter([Candidate(provider, model)])
    return ModelRouter(
        build_candidates(provider, model, keys, options, jobs=jobs),
        balance_width=PROVIDERS[provider].balance_width if options.balance and jobs > 1 else 1,
        fallback=options.fallback,
    )


def switch_reason(exc: DetectorError) -> str:
    if isinstance(exc, RateLimited):
        return "rate limited"
    if isinstance(exc, ParseError):
        return "unparseable reply"
    if exc.status == 404:
        return "model unavailable"
    if exc.status == 504:
        return "timed out"
    if exc.status in (401, 402, 403):
        return "key refused"
    return "provider error"


def run_job(
    job: Job,
    index: int,
    total: int,
    *,
    router: ModelRouter,
    primary: str,
    keys: dict,
    deadline: float,
    cancel: threading.Event,
) -> JobResult:
    tried: set[Candidate] = set()
    failed: Candidate | None = None
    last_error: DetectorError | None = None
    usage: dict = {}

    while True:
        if cancel.is_set():
            raise DetectorError("Request cancelled.", status=499)
        candidate = router.pick(tried, slot=index - 1)
        if candidate is None:
            raise last_error or DetectorError("No model is available.", status=503)
        if failed is not None and last_error is not None:
            router.record_switch(failed, candidate, switch_reason(last_error))

        retries = 0 if router.has_alternative(tried | {candidate}) else 2
        try:
            raw = client.request_chunk(
                job.messages,
                keys[candidate.provider],
                candidate.model,
                candidate.provider,
                rate_limit_retries=retries,
                deadline=deadline,
                max_tokens=job.budget(candidate),
                options=catalogue.request_options(candidate.provider, candidate.model),
                temperature=job.temperature,
                usage=usage,
            )
            value, warnings = job.parse(raw)
            return JobResult(value, warnings, candidate, usage)
        except ParseError as exc:
            # Another job may parse fine on this model; only skip it here.
            prefix = f"Part {index} of {total} could not be parsed. " if total > 1 else ""
            last_error = ParseError(prefix + exc.message, detail=exc.detail)
        except ModelFailure as exc:
            last_error = exc
            router.retire(candidate, exc)
        except DetectorError as exc:
            # A bad key or plan on the *selected* provider is the user's to
            # fix; on a fallback provider it just removes that candidate.
            if candidate.provider == primary or exc.status == 504:
                raise
            last_error = exc
            router.retire(candidate, exc)
        if not router.fallback:
            raise last_error
        tried.add(candidate)
        failed = candidate


def run_jobs(
    jobs: list[Job],
    *,
    router: ModelRouter,
    primary: str,
    keys: dict,
    options: ScanOptions,
    deadline: float | None = None,
) -> list[JobResult]:
    """Run jobs in parallel; results come back in job order."""
    if not jobs:
        return []
    deadline = deadline or time.monotonic() + config.SCAN_DEADLINE_S
    cancel = threading.Event()
    total = len(jobs)

    def run(item: tuple[int, Job]) -> JobResult:
        index, job = item
        return run_job(
            job, index, total, router=router, primary=primary, keys=keys,
            deadline=deadline, cancel=cancel,
        )

    workers = options.concurrency or PROVIDERS[primary].concurrency
    workers = max(1, min(workers, MAX_CONCURRENCY, total))
    numbered = list(enumerate(jobs, start=1))
    if workers == 1:
        return [run(item) for item in numbered]
    pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="model")
    try:
        futures = [pool.submit(run, item) for item in numbered]
        return [f.result() for f in futures]
    except BaseException:
        cancel.set()
        raise
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def summarize(results: list[JobResult], router: ModelRouter) -> dict:
    """Usage, models used, and fallback switches for a run."""
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    used: dict[Candidate, int] = {}
    for result in results:
        used[result.candidate] = used.get(result.candidate, 0) + 1
        for name in usage:
            usage[name] += result.usage.get(name, 0)
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
    ranked = sorted(used.items(), key=lambda kv: -kv[1])
    return {
        "usage": usage,
        "main": ranked[0][0] if ranked else None,
        "models_used": [
            {"provider": c.provider, "model": c.model, "jobs": n} for c, n in ranked
        ],
        "fallbacks": router.switches,
    }
