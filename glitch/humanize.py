"""Humanize: rewrite AI-sounding sentences so they read naturally.

Token-lean by design:

* only sentences labelled AI or AI-polished are rewritten (labels come from
  the last check, or from the quick-mode detector, which is cached);
* each target travels with at most one neighbour on each side as read-only
  context, never the whole document;
* the model returns only the rewritten sentences, keyed by number;
* rewrites are cached per sentence + tone + selected model, so repeating or
  editing a request re-sends only what changed;
* output budgets are sized to the targets, via the shared router (fallback,
  balancing, low reasoning effort).

After rewriting, the result is re-scored with the quick detector (only the
new sentences cost tokens). An optional second pass rewrites sentences that
still look AI-written.
"""

from __future__ import annotations

import time

from . import catalogue, config
from .analyzer import REASONING_HEADROOM, label_sentences
from .cache import TTLCache, fingerprint
from .errors import DetectorError
from .parsing import parse_rewrites
from .prompt import HUMANIZE_VERSION, TONES, build_humanize_messages
from .providers import PROVIDERS, get_provider, validate_key, validate_model
from .routing import Candidate, Job, ScanOptions, make_router, run_jobs, summarize
from .text import document_sentences, replace_sentences

FLAGGED = ("AI", "REFINED")
GROUP_WORDS = 350          # target words per request: keeps replies short and reliable
REWRITE_TEMPERATURE = 0.7  # rewriting needs variety; detection runs at 0
REWRITE_CACHE = TTLCache(5_000, 6 * 3600)


def _budget(candidate: Candidate, target_chars: int) -> int:
    budget = 64 + int(target_chars / 3.2 * 1.4)  # rewritten text ≈ 1.0–1.3x + JSON
    if catalogue.is_reasoning(candidate.provider, candidate.model):
        budget += REASONING_HEADROOM
    return max(128, budget)


def _groups(targets: list[int], sentences: list[str]) -> list[list[int]]:
    groups: list[list[int]] = []
    words = 0
    for index in targets:
        count = len(sentences[index].split())
        if groups and words + count <= GROUP_WORDS:
            groups[-1].append(index)
            words += count
        else:
            groups.append([index])
            words = count
    return groups


def _job(group: list[int], sentences: list[str], tone: str, scope: tuple) -> Job:
    targets = set(group)
    shown: list[int] = []
    for index in group:
        for neighbour in (index - 1, index, index + 1):
            if 0 <= neighbour < len(sentences) and neighbour not in shown:
                shown.append(neighbour)
    shown.sort()
    items = [(i + 1 if i in targets else None, sentences[i]) for i in shown]
    numbers = [i + 1 for i in group]

    def parse(raw: str):
        rewrites, warnings = parse_rewrites(raw, numbers)
        out = {n - 1: text for n, text in rewrites.items()}
        for index, text in out.items():
            REWRITE_CACHE.put((*scope, fingerprint(sentences[index])), text)
        return out, warnings

    target_chars = sum(len(sentences[i]) for i in group)
    return Job(
        messages=build_humanize_messages(items, tone),
        parse=parse,
        budget=lambda c: _budget(c, target_chars),
        temperature=REWRITE_TEMPERATURE,
    )


def _rewrite(
    indices: list[int],
    sentences: list[str],
    *,
    provider: str,
    model: str,
    keys: dict,
    tone: str,
    options: ScanOptions,
    deadline: float,
) -> tuple[dict[int, str], dict, int]:
    """Rewrite sentences at ``indices``; returns (rewrites, run info, cache hits)."""
    scope = (HUMANIZE_VERSION, provider, model, tone)
    rewrites: dict[int, str] = {}
    pending: list[int] = []
    for index in indices:
        hit = REWRITE_CACHE.get((*scope, fingerprint(sentences[index])))
        if hit is None:
            pending.append(index)
        else:
            rewrites[index] = hit

    groups = _groups(pending, sentences)
    router = make_router(provider, model, keys, options, jobs=len(groups))
    jobs = [_job(g, sentences, tone, scope) for g in groups]
    results = run_jobs(
        jobs, router=router, primary=provider, keys=keys, options=options, deadline=deadline
    )
    warnings: list[str] = []
    for result in results:
        rewrites.update(result.value)
        warnings += [w for w in result.warnings if w not in warnings]
    info = summarize(results, router)
    info["warnings"] = warnings
    info["calls"] = len(groups)
    return rewrites, info, len(indices) - len(pending)


def _score(labels: list[str]) -> int:
    flagged = sum(1 for label in labels if label in FLAGGED)
    return int(round(100 * flagged / len(labels))) if labels else 0


def _merge_usage(*infos: dict) -> dict:
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for info in infos:
        for name in usage:
            usage[name] += info.get("usage", {}).get(name, 0)
    return usage


def humanize(
    text: str,
    *,
    provider: str,
    model: str = "",
    keys: dict,
    tone: str = "natural",
    only_flagged: bool = True,
    labels: list[dict] | None = None,
    second_pass: bool = False,
    score_after_rewrite: bool = True,
    options: ScanOptions | None = None,
) -> dict:
    started = time.monotonic()
    deadline = started + config.SCAN_DEADLINE_S
    options = options or ScanOptions()
    spec = get_provider(provider)
    keys = {p: (keys or {}).get(p, "").strip() for p in PROVIDERS}
    tone = tone if tone in TONES else "natural"

    text = (text or "").strip()
    if not text:
        raise DetectorError("Paste some text to rewrite.", status=400)
    validate_key(spec.id, keys[spec.id])
    model = (model or "").strip() or spec.default_model
    validate_model(spec.id, model)
    route = dict(provider=spec.id, model=model, keys=keys, options=options, deadline=deadline)

    sentences = document_sentences(text)
    if not sentences:
        raise DetectorError("Paste a few full sentences to rewrite.", status=400)

    # 1. Labels: reuse the client's last check; label only what's missing.
    known = {
        str(item.get("text", "")): str(item.get("classification", "")).upper()
        for item in (labels or [])
        if isinstance(item, dict)
    }
    infos: list[dict] = []
    if only_flagged:
        missing = [s for s in sentences if s not in known]
        if missing:
            fresh, info = label_sentences(missing, **route)
            known.update(fresh)
            infos.append(info)
        before = [known.get(s, "HUMAN") for s in sentences]
        targets = [i for i, label in enumerate(before) if label in FLAGGED]
    else:
        before = [known.get(s, "") for s in sentences]
        targets = list(range(len(sentences)))
    score_before = _score(before) if all(before) else None

    if not targets:
        return {
            "text": text,
            "html": replace_sentences(text, sentences, {})[1],
            "changes": [],
            "rewritten": 0,
            "targets": 0,
            "reused": 0,
            "score_before": score_before,
            "score_after": score_before,
            "tone": tone,
            "usage": _merge_usage(*infos),
            "warnings": [],
            "fallbacks": [],
            "models_used": [],
            "elapsed_ms": int((time.monotonic() - started) * 1000),
        }

    # 2. Rewrite flagged sentences (cache first).
    rewrites, rewrite_info, reused = _rewrite(targets, sentences, tone=tone, **route)
    infos.append(rewrite_info)
    warnings = list(rewrite_info["warnings"])
    new_text, html = replace_sentences(text, sentences, rewrites)

    # 3. Re-score. Unchanged sentences keep their known labels; only the
    #    rewritten ones are sent (and the client's labels are never cached).
    def rescore(new_text: str) -> int:
        new_sentences = document_sentences(new_text)
        missing = [s for s in dict.fromkeys(new_sentences) if s not in known]
        if missing:
            fresh, info = label_sentences(missing, **route)
            known.update(fresh)
            infos.append(info)
        return _score([known.get(s, "HUMAN") for s in new_sentences])

    score_after = None
    if score_after_rewrite or second_pass:
        score_after = rescore(new_text)
        # Optionally give rewrites that still look AI-written one more pass.
        still = [i for i, s in rewrites.items() if known.get(s) in FLAGGED] if second_pass else []
        if still:
            current = list(sentences)
            for i in still:
                current[i] = rewrites[i]
            again, info, _ = _rewrite(still, current, tone=tone, **route)
            infos.append(info)
            rewrites.update(again)
            new_text, html = replace_sentences(text, sentences, rewrites)
            score_after = rescore(new_text)

    if len(rewrites) < len(targets):
        unchanged = len(targets) - len(rewrites)
        note = f"{unchanged} flagged sentence(s) were left as they were."
        if note not in warnings:
            warnings.append(note)

    fallbacks = [s for i in infos for s in i.get("fallbacks", [])]
    return {
        "text": new_text,
        "html": html,
        "changes": [
            {"index": i, "before": sentences[i], "after": rewrites[i]} for i in sorted(rewrites)
        ],
        "rewritten": len(rewrites),
        "targets": len(targets),
        "reused": reused,
        "score_before": score_before,
        "score_after": score_after,
        "tone": tone,
        "usage": _merge_usage(*infos),
        "warnings": warnings,
        "fallbacks": fallbacks,
        "models_used": rewrite_info["models_used"],
        "elapsed_ms": int((time.monotonic() - started) * 1000),
    }
