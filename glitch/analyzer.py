"""Scan orchestration: chunking, parallel calls, routing, caching, scoring.

All chunks of one scan share a ``ModelRouter``:

* **Load balancing** — chunks are spread round-robin over the provider's top
  healthy models, multiplying per-model rate-limit budgets.
* **Fallback** — when a model is rate limited (or gone, or failing), it is
  retired for the rest of the scan and chunks move on immediately instead of
  sleeping on ``Retry-After``. Rate-limited models also enter a process-wide
  cooldown so the next scan starts on a model likely to answer.

Finished chunks are cached in memory, so re-checking unchanged text is free.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import Counter, OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from . import catalogue, client, config
from .errors import DetectorError, ModelFailure, ParseError, RateLimited
from .parsing import parse_analysis, parse_labels
from .prompt import PROMPT_VERSION, build_compact_messages, build_messages
from .providers import PROVIDERS, get_provider, validate_key, validate_model
from .text import LABELS, Sentence, build_highlight_html, chunk_text, split_sentences

DEFAULT_COOLDOWN_S = 60
MAX_CONCURRENCY = 4
CROSS_PROVIDER_MODELS = 2
REASONING_HEADROOM = 768  # hidden thinking tokens, even at low effort
CACHE_SIZE = 512
CACHE_TTL_S = 3600

_cooldowns: dict[tuple[str, str], float] = {}
_cooldown_lock = threading.Lock()


def cooldown_remaining(provider: str, model: str) -> float:
    with _cooldown_lock:
        return max(0.0, _cooldowns.get((provider, model), 0.0) - time.monotonic())


def _start_cooldown(provider: str, model: str, seconds: float) -> None:
    with _cooldown_lock:
        _cooldowns[(provider, model)] = time.monotonic() + (seconds or DEFAULT_COOLDOWN_S)


@dataclass(frozen=True)
class Candidate:
    provider: str
    model: str

    def __str__(self) -> str:
        return f"{PROVIDERS[self.provider]['label']} · {self.model}"


@dataclass
class ScanOptions:
    chunk_words: int = 900
    fallback: bool = True
    cross_provider: bool = True
    concurrency: int = 0  # 0 = provider default
    compact: bool = True
    balance: bool = True


@dataclass
class ChunkResult:
    sentences: list[Sentence]
    warnings: list[str]
    candidate: Candidate
    cached: bool = False
    usage: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Result cache
# ---------------------------------------------------------------------------
_cache: OrderedDict[tuple, tuple[float, list[Sentence], list[str]]] = OrderedDict()
_cache_lock = threading.Lock()


def _cache_key(candidate: Candidate, compact: bool, chunk: str) -> tuple:
    digest = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
    return (PROMPT_VERSION, compact, candidate.provider, candidate.model, digest)


def _cache_get(key: tuple):
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        if time.monotonic() - entry[0] > CACHE_TTL_S:
            del _cache[key]
            return None
        _cache.move_to_end(key)
        return entry[1], entry[2]


def _cache_put(key: tuple, sentences: list[Sentence], warnings: list[str]) -> None:
    with _cache_lock:
        _cache[key] = (time.monotonic(), sentences, warnings)
        _cache.move_to_end(key)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)


def clear_cache() -> None:
    with _cache_lock:
        _cache.clear()


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------
class ModelRouter:
    """Thread-safe model choice shared by every chunk of one scan."""

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
        """Best candidate for a chunk; ``slot`` spreads chunks across models."""
        with self._lock:
            usable = self._usable(tried)
        if not usable:
            return None
        fresh = [c for c in usable if not cooldown_remaining(c.provider, c.model)]
        pool = fresh or usable
        if self._balance_width > 1:
            # Balance only within the selected provider; the other provider
            # stays a fallback.
            spread = [c for c in pool if c.provider == self._primary][: self._balance_width]
            if spread:
                return spread[slot % len(spread)]
        return pool[0]

    def has_alternative(self, tried: set[Candidate]) -> bool:
        with self._lock:
            return len(self._usable(tried)) > 0

    def retire(self, candidate: Candidate, exc: DetectorError) -> None:
        if isinstance(exc, RateLimited):
            _start_cooldown(candidate.provider, candidate.model, exc.retry_after)
        with self._lock:
            self._retired.add(candidate)

    def record_switch(self, src: Candidate, dst: Candidate, reason: str) -> None:
        with self._lock:
            if not any(s["from"] == str(src) and s["to"] == str(dst) for s in self.switches):
                self.switches.append({"from": str(src), "to": str(dst), "reason": reason})


def build_candidates(
    provider: str, model: str, keys: dict, options: ScanOptions, *, chunks: int = 1
) -> list[Candidate]:
    out = [Candidate(provider, model)]
    if not (options.fallback or (options.balance and chunks > 1)):
        return out

    def add(c: Candidate) -> None:
        if c not in out:
            out.append(c)

    for alternative in catalogue.fallback_models(provider):
        add(Candidate(provider, alternative))
    if options.fallback and options.cross_provider:
        for other in PROVIDERS:
            key = keys.get(other, "")
            if other == provider or len(key) < config.MIN_KEY_LENGTH:
                continue
            for alternative in catalogue.fallback_models(other)[:CROSS_PROVIDER_MODELS]:
                add(Candidate(other, alternative))
    return out


def _switch_reason(exc: DetectorError) -> str:
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


# ---------------------------------------------------------------------------
# One chunk
# ---------------------------------------------------------------------------
def output_budget(candidate: Candidate, chunk: str, sentence_count: int, compact: bool) -> int:
    """A ``max_tokens`` sized to the expected reply instead of a flat 4096.

    Groq counts the requested budget against the tokens-per-minute limit, so
    asking for only what the reply needs lets more calls fit per minute.
    """
    if compact:
        budget = 48 + 8 * sentence_count  # '"REFINED",' ≈ 4 tokens
    else:
        budget = 96 + len(chunk) // 3 + 16 * sentence_count  # echoed text + JSON
    if catalogue.is_reasoning(candidate.provider, candidate.model):
        budget += REASONING_HEADROOM
    return max(128, budget)


def _scan_chunk(
    chunk: str,
    index: int,
    total: int,
    *,
    router: ModelRouter,
    primary: str,
    keys: dict,
    compact: bool,
    deadline: float,
    cancel: threading.Event,
) -> ChunkResult:
    sources = split_sentences(chunk) if compact else []
    if compact:
        messages = build_compact_messages(sources, index, total)
    else:
        messages = build_messages(chunk, index, total)
    sentence_count = len(sources) or len(split_sentences(chunk))

    tried: set[Candidate] = set()
    failed: Candidate | None = None
    last_error: DetectorError | None = None
    usage: dict = {}

    while True:
        if cancel.is_set():
            raise DetectorError("Scan cancelled.", status=499)
        candidate = router.pick(tried, slot=index - 1)
        if candidate is None:
            raise last_error or DetectorError("No free model is available.", status=503)
        if failed is not None and last_error is not None:
            router.record_switch(failed, candidate, _switch_reason(last_error))

        key = _cache_key(candidate, compact, chunk)
        hit = _cache_get(key)
        if hit is not None:
            return ChunkResult(list(hit[0]), list(hit[1]), candidate, cached=True)

        retries = 0 if router.has_alternative(tried | {candidate}) else 2
        try:
            raw = client.request_chunk(
                messages,
                keys[candidate.provider],
                candidate.model,
                candidate.provider,
                rate_limit_retries=retries,
                deadline=deadline,
                max_tokens=output_budget(candidate, chunk, sentence_count, compact),
                options=catalogue.request_options(candidate.provider, candidate.model),
                usage=usage,
            )
            if compact:
                labels, warnings = parse_labels(raw, len(sources))
                sentences = [Sentence(text, label) for text, label in zip(sources, labels)]
            else:
                sentences, warnings = parse_analysis(raw)
            _cache_put(key, sentences, warnings)
            return ChunkResult(sentences, warnings, candidate, usage=usage)
        except ParseError as exc:
            # Another chunk may parse fine on this model; only skip it here.
            last_error = ParseError(
                f"Part {index} of {total} could not be parsed. {exc.message}",
                detail=exc.detail,
            )
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


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def analyze(
    text: str,
    *,
    provider: str = "openrouter",
    model: str = "",
    keys: dict | None = None,
    options: ScanOptions | None = None,
) -> dict:
    started = time.monotonic()
    deadline = started + config.SCAN_DEADLINE_S
    options = options or ScanOptions()
    cfg = get_provider(provider)
    keys = {p: (keys or {}).get(p, "").strip() for p in PROVIDERS}

    text = (text or "").strip()
    if not text:
        raise DetectorError("Paste some text to analyze.", status=400)
    validate_key(cfg["key"], keys[cfg["key"]])
    model = (model or "").strip() or cfg["default_model"]
    validate_model(cfg["key"], model)

    chunks = chunk_text(text, options.chunk_words)
    total_chunks = len(chunks)
    router = ModelRouter(
        build_candidates(cfg["key"], model, keys, options, chunks=total_chunks),
        balance_width=cfg["balance_width"] if options.balance and total_chunks > 1 else 1,
        fallback=options.fallback,
    )
    cancel = threading.Event()

    def run(item: tuple[int, str]) -> ChunkResult:
        index, chunk = item
        return _scan_chunk(
            chunk,
            index,
            total_chunks,
            router=router,
            primary=cfg["key"],
            keys=keys,
            compact=options.compact,
            deadline=deadline,
            cancel=cancel,
        )

    workers = options.concurrency or cfg["concurrency"]
    workers = max(1, min(workers, MAX_CONCURRENCY, total_chunks))
    jobs = list(enumerate(chunks, start=1))
    if workers == 1:
        results = [run(job) for job in jobs]
    else:
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="scan")
        try:
            futures = [pool.submit(run, job) for job in jobs]
            results = [f.result() for f in futures]
        except BaseException:
            cancel.set()
            raise
        finally:
            pool.shutdown(wait=False, cancel_futures=True)

    sentences: list[Sentence] = []
    warnings: list[str] = []
    used: Counter[Candidate] = Counter()
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    for result in results:
        sentences.extend(result.sentences)
        used[result.candidate] += 1
        for name in usage:
            usage[name] += result.usage.get(name, 0)
        for warning in result.warnings:
            if warning not in warnings:
                warnings.append(warning)
    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

    primary = Candidate(cfg["key"], model)
    if (
        primary not in used
        and not router.switches
        and cooldown_remaining(primary.provider, primary.model)
    ):
        warnings.insert(0, f"{model} is cooling down after a rate limit; another free model was used.")
    for switch in router.switches:
        warnings.insert(0, f"Switched {switch['from']} → {switch['to']} ({switch['reason']}).")

    counts = {label: 0 for label in LABELS}
    for sentence in sentences:
        counts[sentence.classification] += 1

    total = len(sentences)
    ai_like = counts["AI"] + counts["REFINED"]
    score = int(round(100 * ai_like / total)) if total else 0

    estimated = len(split_sentences(text))
    if (
        not options.compact
        and estimated
        and abs(estimated - total) > max(2, int(estimated * 0.25))
    ):
        warnings.append(
            f"The model returned {total} sentences; the source appears to contain about {estimated}. "
            "Sentence boundaries may have been merged or split."
        )

    main = used.most_common(1)[0][0]
    return {
        "sentences": [
            {"text": s.text, "classification": s.classification} for s in sentences
        ],
        "counts": counts,
        "score": score,
        "total": total,
        "ai_like": ai_like,
        "estimated_total": estimated,
        "chunks": total_chunks,
        "cached_chunks": sum(1 for r in results if r.cached),
        "mode": "compact" if options.compact else "classic",
        "usage": usage,
        "warnings": warnings,
        "html": build_highlight_html(sentences, text),
        "model": main.model,
        "provider": main.provider,
        "provider_label": PROVIDERS[main.provider]["label"],
        "requested_model": model,
        "models_used": [
            {"provider": c.provider, "model": c.model, "chunks": n}
            for c, n in used.most_common()
        ],
        "fallbacks": router.switches,
        "words": len(text.split()),
        "chars": len(text),
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }
