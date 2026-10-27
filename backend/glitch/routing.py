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
from .limits import ledger
from .providers import PAID, PROVIDERS

DEFAULT_COOLDOWN_S = 60
UNAVAILABLE_COOLDOWN_S = 3600  # "model not found": skip it for an hour, not just this run
MAX_CONCURRENCY = 4
CROSS_PROVIDER_MODELS = 2
MAX_ATTEMPTS = 12        # per job, across models and waits
WAIT_MARGIN_S = 15       # keep this much of the deadline for the request itself
MAX_PATIENT_WAIT_S = 65  # wait out a per-minute window; daily caps fail fast

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
    top_p: float = 1.0
    # Extra sampling parameters per provider id (e.g. {"openrouter": {"min_p": 0.05}}).
    # Sent as optional options: dropped automatically if a provider rejects them.
    provider_options: dict = field(default_factory=dict)

    def prompt_tokens(self) -> int:
        """Rough input size (≈4 characters per token) for rate-limit pacing."""
        return sum(len(str(m.get("content", ""))) for m in self.messages) // 4


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

    @staticmethod
    def wait_for(candidate: Candidate, tokens: int) -> float:
        """Seconds until ``candidate`` can take ~``tokens`` (cooldowns, daily caps, pacing)."""
        return max(
            cooldown_remaining(candidate.provider, candidate.model),
            ledger.wait_for(candidate.provider, candidate.model, tokens),
        )

    def pick(self, tried: set[Candidate], slot: int = 0, tokens: int = 0) -> tuple[Candidate | None, float]:
        """Best candidate for a job and how long to wait for it.

        Models that can take the request now are preferred (``slot`` spreads
        jobs across the selected provider's top models). If every model is
        busy, the one that frees up soonest is returned with its wait.
        """
        with self._lock:
            usable = self._usable(tried)
        if not usable:
            return None, 0.0
        waits = {c: self.wait_for(c, tokens) for c in usable}
        ready = [c for c in usable if waits[c] <= 0]
        if not ready:
            soonest = min(usable, key=lambda c: waits[c])
            return soonest, waits[soonest]
        if self._balance_width > 1:
            # Balance only within the selected provider; others stay fallbacks.
            spread = [c for c in ready if c.provider == self._primary][: self._balance_width]
            if spread:
                return spread[slot % len(spread)], 0.0
        return ready[0], 0.0

    def has_alternative(self, tried: set[Candidate]) -> bool:
        """Is another model ready right now? (Cooling ones don't count.)"""
        with self._lock:
            usable = self._usable(tried)
        return any(self.wait_for(c, 0) <= 0 for c in usable)

    def retire(self, candidate: Candidate, exc: DetectorError) -> None:
        """Take a failing model out of rotation.

        Rate limits only cool the model down (the ledger knows until when), so
        it can come back later in the same run; other failures retire it.
        """
        if isinstance(exc, RateLimited):
            start_cooldown(candidate.provider, candidate.model, exc.retry_after)
            return
        if exc.status == 404:
            start_cooldown(candidate.provider, candidate.model, UNAVAILABLE_COOLDOWN_S)
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
    prompt_tokens = job.prompt_tokens()

    for _ in range(MAX_ATTEMPTS):
        if cancel.is_set():
            raise DetectorError("Request cancelled.", status=499)
        candidate, wait = router.pick(tried, slot=index - 1, tokens=prompt_tokens + 512)
        if candidate is None:
            raise last_error or DetectorError("No model is available.", status=503)
        if wait > 0:
            # Every model is busy: wait for the soonest one when it frees up
            # shortly (per-minute windows) and the deadline allows, instead of
            # failing the whole request. Daily caps and long resets fail fast.
            if (
                not router.fallback
                or wait > MAX_PATIENT_WAIT_S
                or time.monotonic() + wait + WAIT_MARGIN_S > deadline
            ):
                if last_error is not None and not isinstance(last_error, RateLimited):
                    raise last_error
                raise RateLimited(
                    "Every free model is busy right now. Try again in about "
                    f"{max(1, round(wait / 60))} minute(s), or check a shorter text.",
                    detail=last_error.message if last_error else "",
                    retry_after=wait,
                )
            if cancel.wait(wait):
                raise DetectorError("Request cancelled.", status=499)
        if failed is not None and last_error is not None and candidate != failed:
            router.record_switch(failed, candidate, switch_reason(last_error))

        budget = job.budget(candidate)
        ledger.spend(candidate.provider, candidate.model, prompt_tokens + budget)
        retries = 0 if router.has_alternative(tried | {candidate}) else 2
        try:
            raw = client.request_chunk(
                job.messages,
                keys[candidate.provider],
                candidate.model,
                candidate.provider,
                rate_limit_retries=retries,
                deadline=deadline,
                max_tokens=budget,
                options={
                    **catalogue.request_options(candidate.provider, candidate.model),
                    **job.provider_options.get(candidate.provider, {}),
                },
                temperature=job.temperature,
                top_p=job.top_p,
                usage=usage,
            )
            value, warnings = job.parse(raw)
            return JobResult(value, warnings, candidate, usage)
        except ParseError as exc:
            # Another job may parse fine on this model; only skip it here.
            prefix = f"Part {index} of {total} could not be parsed. " if total > 1 else ""
            last_error = ParseError(prefix + exc.message, detail=exc.detail)
        except RateLimited as exc:
            # Cooled down, not retired: this job may come back to it later.
            last_error = exc
            router.retire(candidate, exc)
            failed = candidate
            if not router.fallback:
                raise
            continue
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
    raise last_error or DetectorError("No model is available.", status=503)


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
