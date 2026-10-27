"""AI-writing detection: build jobs, run them through ``routing``, score.

In quick (compact) mode every sentence's label is cached under the user's
selected provider + model, so an edited document only sends its new or
changed sentences. Classic mode caches whole sections.
"""

from __future__ import annotations

import time

from . import catalogue, config
from .cache import TTLCache, fingerprint
from .errors import DetectorError
from .parsing import parse_analysis, parse_labels
from .prompt import PROMPT_VERSION, build_compact_messages, build_messages
from .providers import PROVIDERS, get_provider, validate_key, validate_model
from .routing import (
    Candidate,
    Job,
    ScanOptions,
    cooldown_remaining,
    make_router,
    run_jobs,
    summarize,
)
from .text import (
    LABELS,
    Sentence,
    build_highlight_html,
    chunk_text,
    document_sentences,
    group_sentences,
    normalize_text,
    split_sentences,
)

__all__ = ["Candidate", "ScanOptions", "analyze", "output_budget", "label_sentences"]

REASONING_HEADROOM = 768  # hidden thinking tokens, even at low effort
CACHE_TTL_S = 6 * 3600

# Keys are scoped to what the user *selected* (provider + model + mode), not to
# whichever model balancing or fallback used, so re-checks always hit.
LABEL_CACHE = TTLCache(20_000, CACHE_TTL_S)  # sentence -> label
CHUNK_CACHE = TTLCache(512, CACHE_TTL_S)     # classic section -> (sentences, warnings)


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


def _compact_job(sources: list[str], index: int, total: int, scope: tuple) -> Job:
    def parse(raw: str):
        labels, warnings = parse_labels(raw, len(sources))
        # Cached as soon as each group finishes, so a run that fails later
        # still keeps the work done so far.
        for text, label in zip(sources, labels):
            LABEL_CACHE.put((*scope, fingerprint(text)), label)
        return labels, warnings

    joined = "\n".join(sources)
    return Job(
        messages=build_compact_messages(sources, index, total),
        parse=parse,
        budget=lambda c: output_budget(c, joined, len(sources), True),
    )


def _classic_job(chunk: str, index: int, total: int, scope: tuple) -> Job:
    def parse(raw: str):
        sentences, warnings = parse_analysis(raw)
        CHUNK_CACHE.put((*scope, fingerprint(chunk)), (sentences, warnings))
        return sentences, warnings

    count = len(split_sentences(chunk))
    return Job(
        messages=build_messages(chunk, index, total),
        parse=parse,
        budget=lambda c: output_budget(c, chunk, count, False),
    )


def label_sentences(
    sentences: list[str],
    *,
    provider: str,
    model: str,
    keys: dict,
    options: ScanOptions,
    deadline: float | None = None,
) -> tuple[dict[str, str], dict]:
    """Quick-mode labels for sentences, sending only uncached ones.

    Returns ``(labels by sentence, run info)``. Shared with humanize, which
    re-scores its rewrites this way.
    """
    scope = (PROMPT_VERSION, "compact", provider, model)
    labels: dict[str, str] = {}
    pending: list[str] = []
    for sentence in dict.fromkeys(sentences):  # unique, in order
        hit = LABEL_CACHE.get((*scope, fingerprint(sentence)))
        if hit is None:
            pending.append(sentence)
        else:
            labels[sentence] = hit

    groups = group_sentences(pending, options.chunk_words)
    router = make_router(provider, model, keys, options, jobs=len(groups))
    jobs = [_compact_job(g, i, len(groups), scope) for i, g in enumerate(groups, start=1)]
    results = run_jobs(
        jobs, router=router, primary=provider, keys=keys, options=options, deadline=deadline
    )
    warnings: list[str] = []
    for group, result in zip(groups, results):
        labels.update(zip(group, result.value))
        warnings += [w for w in result.warnings if w not in warnings]

    info = summarize(results, router)
    pending_set = set(pending)
    info.update(
        calls=len(groups),
        reused=sum(1 for s in sentences if s not in pending_set),
        warnings=warnings,
    )
    return labels, info


def _classic(text: str, *, provider: str, model: str, keys: dict, options: ScanOptions):
    scope = (PROMPT_VERSION, "classic", provider, model)
    chunks = chunk_text(text, options.chunk_words)
    cached = {i: CHUNK_CACHE.get((*scope, fingerprint(c))) for i, c in enumerate(chunks)}
    todo = [i for i, hit in cached.items() if hit is None]

    router = make_router(provider, model, keys, options, jobs=len(todo))
    jobs = [_classic_job(chunks[i], n, len(todo), scope) for n, i in enumerate(todo, start=1)]
    results = run_jobs(jobs, router=router, primary=provider, keys=keys, options=options)
    fresh = dict(zip(todo, results))

    sentences: list[Sentence] = []
    warnings: list[str] = []
    for i in range(len(chunks)):
        found, chunk_warnings = (fresh[i].value, fresh[i].warnings) if i in fresh else cached[i]
        sentences += found
        warnings += [w for w in chunk_warnings if w not in warnings]

    info = summarize(results, router)
    info.update(calls=len(todo), cached_chunks=len(chunks) - len(todo), reused=0, warnings=warnings)
    return sentences, info


def analyze(
    text: str,
    *,
    provider: str = "openrouter",
    model: str = "",
    keys: dict | None = None,
    options: ScanOptions | None = None,
) -> dict:
    started = time.monotonic()
    options = options or ScanOptions()
    spec = get_provider(provider)
    keys = {p: (keys or {}).get(p, "").strip() for p in PROVIDERS}

    # Detectors see through invisible characters and lookalike letters; so do we.
    text = normalize_text(text).strip()
    if not text:
        raise DetectorError("Paste some text to analyze.", status=400)
    validate_key(spec.id, keys[spec.id])
    model = (model or "").strip() or spec.default_model
    validate_model(spec.id, model)

    if options.compact:
        all_sentences = document_sentences(text)
        labels, info = label_sentences(
            all_sentences, provider=spec.id, model=model, keys=keys, options=options,
            deadline=started + config.SCAN_DEADLINE_S,
        )
        sentences = [Sentence(s, labels[s]) for s in all_sentences]
        info.setdefault("cached_chunks", 0)
    else:
        sentences, info = _classic(text, provider=spec.id, model=model, keys=keys, options=options)

    warnings = list(info["warnings"])
    primary = Candidate(spec.id, model)
    if (
        info["main"] not in (None, primary)
        and not info["fallbacks"]
        and cooldown_remaining(primary.provider, primary.model)
    ):
        warnings.insert(0, f"{model} is cooling down after a rate limit; another model was used.")
    for switch in info["fallbacks"]:
        warnings.insert(0, f"Switched {switch['from']} → {switch['to']} ({switch['reason']}).")

    counts = {label: 0 for label in LABELS}
    for sentence in sentences:
        counts[sentence.classification] += 1
    total = len(sentences)
    ai_like = counts["AI"] + counts["REFINED"]
    score = int(round(100 * ai_like / total)) if total else 0

    estimated = len(split_sentences(text))
    if not options.compact and estimated and abs(estimated - total) > max(2, int(estimated * 0.25)):
        warnings.append(
            f"The model returned {total} sentences; the source appears to contain about {estimated}. "
            "Sentence boundaries may have been merged or split."
        )

    main = info["main"] or primary
    return {
        "sentences": [{"text": s.text, "classification": s.classification} for s in sentences],
        "counts": counts,
        "score": score,
        "total": total,
        "ai_like": ai_like,
        "estimated_total": estimated,
        "chunks": info["calls"],
        "cached_chunks": info["cached_chunks"],
        "reused_sentences": info["reused"],
        "mode": "compact" if options.compact else "classic",
        "usage": info["usage"],
        "warnings": warnings,
        "html": build_highlight_html(sentences, text),
        "model": main.model,
        "provider": main.provider,
        "provider_label": PROVIDERS[main.provider].label,
        "requested_model": model,
        "models_used": info["models_used"],
        "fallbacks": info["fallbacks"],
        "words": len(text.split()),
        "chars": len(text),
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }
