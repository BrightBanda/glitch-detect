"""Humanize: rewrite AI-sounding passages so they read as one person's writing.

Design follows mechanics-of-ai-writing-detection.md. Detectors lean on rhythm
and discourse shape (T2) and on missing particulars (T3) far more than on
single words (T1), and sentence-by-sentence paraphrase makes text *more*
uniform (§4.4.3). So:

1. **Passages, not sentences.** Each run of consecutive flagged sentences in a
   paragraph is rewritten as one unit, free to merge, split, reorder and cut —
   the only way to change sentence-length rhythm and redundancy.
2. **Specific notes.** ``style.rewrite_notes`` tells the model exactly what
   this passage does wrong (flat rhythm, these words, this tricolon…).
3. **Free validation.** ``style.check_rewrite`` checks each rewrite locally.
   Lost numbers/names → retry, then keep the original if still lost.
   Remaining AI patterns → one retry with that feedback.
4. **Honest grounding.** No invented facts: ``style.invented`` rejects new
   names, numbers and made-up experiences. The writer can answer optional
   questions (``style.grounding_questions``) and those *facts* are the only new
   material allowed (§2.5, §2.8: the one thing that moves the deepest signals
   is authorship).
5. **Best of two.** Passages that still read as AI get two differently-built
   versions in one request; the better one is chosen locally (facts first,
   then style) and only the winner is re-scored.

Changes follow reports/How AI humanizers work.md (Phases 0, 1, 3 and 4).

Token-lean: only flagged passages travel, each with one neighbouring sentence
of context; replies contain only rewritten passages; results are cached per
passage + tone + model; re-scoring sends only new sentences.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from . import catalogue, config, style
from .analyzer import REASONING_HEADROOM, label_sentences
from .cache import TTLCache, fingerprint
from .errors import DetectorError
from .parsing import parse_rewrites
from .prompt import HUMANIZE_VERSION, TONES, VOICES, build_humanize_messages
from .providers import PROVIDERS, get_provider, validate_key, validate_model
from .routing import Candidate, Job, ScanOptions, make_router, run_jobs, summarize
from .text import (
    document_sentences,
    locate_sentences,
    normalize_text,
    paragraph_sentences,
    replace_spans,
)

FLAGGED = ("AI", "REFINED")
GROUP_WORDS = 400           # passage words per request: keeps replies short and reliable
# Sampling: higher temperature varies word choice (low temperature is what
# likelihood detectors find easiest); top_p / min_p keep it coherent.
# Detection runs at 0. Heuristic starting values: compare with eval/.
REWRITE_TEMPERATURE = 1.0
REWRITE_TOP_P = 0.95
REWRITE_PROVIDER_OPTIONS = {"openrouter": {"min_p": 0.05}}
REWRITE_CACHE = TTLCache(5_000, 6 * 3600)


@dataclass
class Passage:
    number: int                 # 1-based id used in the prompt and reply
    indices: list[int]          # consecutive sentence indices in the document
    text: str
    before: str | None = None   # neighbouring sentences, read-only context
    after: str | None = None
    notes: list[str] = field(default_factory=list)
    facts: str = ""             # author-supplied details for this passage's paragraph


def paragraph_key(paragraph: list[str]) -> str:
    """Stable id for a paragraph's grounding answers (survives re-segmentation)."""
    return fingerprint(" ".join(paragraph))[:16]


def grounding_questions(text: str, labels: list[dict] | None = None, limit: int = 4) -> list[dict]:
    """Questions for paragraphs that would rewrite better with the writer's details.

    Zero tokens. With ``labels`` (from a check), only paragraphs with flagged
    sentences are asked about; without, paragraphs that make general claims
    with no names or numbers.
    """
    flagged = {
        normalize_text(str(item.get("text", "")))
        for item in (labels or [])
        if isinstance(item, dict) and str(item.get("classification", "")).upper() in FLAGGED
    }
    found = []
    for paragraph in paragraph_sentences(normalize_text(text)):
        joined = " ".join(paragraph)
        questions = style.grounding_questions(joined)
        if labels is not None:
            relevant = any(s in flagged for s in paragraph)
        else:
            relevant = not any(style.names_and_numbers(joined)) and len(questions) > 1
        if relevant and questions:
            opening = " ".join(joined.split()[:10])
            found.append({"key": paragraph_key(paragraph), "opening": opening, "questions": questions})
        if len(found) >= limit:
            break
    return found


def _passages(
    paragraphs: list[list[str]], targets: set[int], *, whole_paragraphs: bool,
    facts: dict[str, str] | None = None,
) -> list[Passage]:
    sentences = [s for p in paragraphs for s in p]
    doc_rhythm = style.sentence_cv(sentences)
    passages: list[Passage] = []
    offset = 0
    for paragraph in paragraphs:
        run: list[int] = []
        indices = list(range(offset, offset + len(paragraph)))
        # Facts go with the paragraph's first passage only, so they're used once.
        pending_facts = (facts or {}).get(paragraph_key(paragraph), "")
        for index in indices + [None]:
            if index is not None and (whole_paragraphs or index in targets):
                run.append(index)
                continue
            if run:
                text = " ".join(sentences[i] for i in run)
                passages.append(
                    Passage(
                        number=len(passages) + 1,
                        indices=run,
                        text=text,
                        before=sentences[run[0] - 1] if run[0] > 0 else None,
                        after=sentences[run[-1] + 1] if run[-1] + 1 < len(sentences) else None,
                        notes=style.rewrite_notes(text, document_rhythm=doc_rhythm),
                        facts=pending_facts,
                    )
                )
                pending_facts = ""
                run = []
        offset += len(paragraph)
    return passages


def _groups(passages: list[Passage]) -> list[list[Passage]]:
    groups: list[list[Passage]] = []
    words = 0
    for passage in passages:
        count = len(passage.text.split())
        if groups and words + count <= GROUP_WORDS:
            groups[-1].append(passage)
            words += count
        else:
            groups.append([passage])
            words = count
    return groups


def _budget(candidate: Candidate, chars: int, variants: int = 1) -> int:
    budget = 96 + int(chars * variants / 3.2 * 1.5)  # rewritten text ≈ 0.8–1.3x + JSON
    if catalogue.is_reasoning(candidate.provider, candidate.model):
        budget += REASONING_HEADROOM
    return max(160, budget)


def _job(
    group: list[Passage], tone: str, voice: str, extra_notes: dict[int, list[str]], variants: int = 1
) -> Job:
    items: list[tuple[str, str]] = []
    for passage in group:
        if passage.before and (not items or items[-1][1] != passage.before):
            items.append(("context", passage.before))
        items.append((str(passage.number), passage.text))
        notes = passage.notes + extra_notes.get(passage.number, [])
        if notes:
            items.append((f"{passage.number} notes", " ".join(notes)))
        if passage.facts:
            items.append((f"{passage.number} facts", passage.facts))
        if passage.after:
            items.append(("context", passage.after))
    numbers = [p.number for p in group]
    chars = sum(len(p.text) for p in group)
    return Job(
        messages=build_humanize_messages(items, tone, voice, variants=variants),
        parse=lambda raw: parse_rewrites(raw, numbers, variants=variants),
        budget=lambda c: _budget(c, chars, variants),
        temperature=REWRITE_TEMPERATURE,
        top_p=REWRITE_TOP_P,
        provider_options=REWRITE_PROVIDER_OPTIONS,
    )


def _run(
    groups: list[list[Passage]], style_key: tuple[str, str], extra: dict[int, list[str]], route: dict,
    variants: int = 1,
):
    router = make_router(
        route["provider"], route["model"], route["keys"], route["options"], jobs=len(groups)
    )
    jobs = [_job(g, *style_key, extra, variants) for g in groups]
    results = run_jobs(
        jobs,
        router=router,
        primary=route["provider"],
        keys=route["keys"],
        options=route["options"],
        deadline=route["deadline"],
    )
    rewrites: dict = {}
    warnings: list[str] = []
    for result in results:
        rewrites.update(result.value)
        warnings += [w for w in result.warnings if w not in warnings]
    # Never hand back invisible characters or lookalike letters.
    rewrites = {
        n: [normalize_text(t) for t in v] if isinstance(v, list) else normalize_text(v)
        for n, v in rewrites.items()
    }
    return rewrites, summarize(results, router), warnings


def _pick(passage: Passage, versions: list[str], voice: str) -> str:
    """Best of several rewrites: facts intact first, then fewest style problems,
    then sentence rhythm closest to the human range."""
    middle = sum(style.HUMAN_CV) / 2

    def rank(text: str) -> tuple:
        facts, problems = style.check_rewrite(passage.text, text, voice=voice, allowed=passage.facts)
        cv = style.sentence_cv(style.split_sentences(text))
        return (len(facts), len(problems), abs((cv if cv is not None else 0.0) - middle))

    return min(versions, key=rank)


def _cache_key(scope: tuple, passage: Passage) -> tuple:
    # Facts are part of the key: a rewrite made without them is never served
    # to a request that has them.
    return (*scope, fingerprint(passage.text), fingerprint(passage.facts) if passage.facts else "")


def _rewrite(
    passages: list[Passage],
    *,
    tone: str,
    voice: str,
    route: dict,
    fresh: bool = False,
    extra_notes: dict[int, list[str]] | None = None,
    variants: int = 1,
) -> tuple[dict[int, str], list[dict], int, list[str]]:
    """Rewrite passages with validation. Returns (rewrites, run infos, cache hits, warnings)."""
    scope = (HUMANIZE_VERSION, route["provider"], route["model"], tone, voice)
    extra = dict(extra_notes or {})
    rewrites: dict[int, str] = {}
    pending: list[Passage] = []
    for passage in passages:
        hit = None if fresh else REWRITE_CACHE.get(_cache_key(scope, passage))
        if hit is None:
            pending.append(passage)
        else:
            rewrites[passage.number] = hit
    reused = len(passages) - len(pending)

    infos: list[dict] = []
    warnings: list[str] = []
    if pending:
        found, info, found_warnings = _run(_groups(pending), (tone, voice), extra, route, variants)
        infos.append(info)
        warnings += found_warnings
        if variants > 1:
            by_number = {p.number: p for p in pending}
            found = {n: _pick(by_number[n], v, voice) for n, v in found.items() if n in by_number}

        # Free local validation; one targeted retry for passages that fail it.
        retry: dict[int, list[str]] = {}
        for passage in pending:
            if passage.number not in found:
                continue
            lost, problems = style.check_rewrite(
                passage.text, found[passage.number], voice=voice, allowed=passage.facts
            )
            if lost or problems:
                retry[passage.number] = lost + problems
        if retry:
            again = [p for p in pending if p.number in retry]
            notes = {n: extra.get(n, []) + ["Your last attempt needs fixing:"] + retry[n] for n in retry}
            # Two versions per retry: far likelier that one keeps every fact.
            second, info, more = _run(_groups(again), (tone, voice), notes, route, variants=2)
            infos.append(info)
            warnings += [w for w in more if w not in warnings]
            by_number = {p.number: p for p in again}
            found.update({n: _pick(by_number[n], v, voice) for n, v in second.items() if n in by_number})

        protected = 0
        for passage in pending:
            new = found.get(passage.number)
            if new is None:
                continue
            lost, _ = style.check_rewrite(passage.text, new, allowed=passage.facts)
            if lost:
                protected += 1  # keep the original rather than drop or invent facts
                continue
            rewrites[passage.number] = new
            REWRITE_CACHE.put(_cache_key(scope, passage), new)
        if protected:
            warnings.append(
                f"Kept the original wording for {protected} passage(s) so no facts were lost or made up."
            )
    return rewrites, infos, reused, warnings


def _score(labels: list[str]) -> int:
    flagged = sum(1 for label in labels if label in FLAGGED)
    return int(round(100 * flagged / len(labels))) if labels else 0


def _merge_usage(infos: list[dict]) -> dict:
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
    voice: str = "personal",
    only_flagged: bool = True,
    labels: list[dict] | None = None,
    second_pass: bool = True,
    score_after_rewrite: bool = True,
    fresh: bool = False,
    options: ScanOptions | None = None,
    facts: dict[str, str] | None = None,
) -> dict:
    started = time.monotonic()
    options = options or ScanOptions()
    spec = get_provider(provider)
    keys = {p: (keys or {}).get(p, "").strip() for p in PROVIDERS}
    tone = tone if tone in TONES else "natural"
    voice = voice if voice in VOICES else "personal"

    text = normalize_text(text).strip()
    if not text:
        raise DetectorError("Paste some text to rewrite.", status=400)
    facts = {
        str(k): normalize_text(str(v)).strip()[:600]
        for k, v in (facts or {}).items()
        if str(v).strip()
    }
    validate_key(spec.id, keys[spec.id])
    model = (model or "").strip() or spec.default_model
    validate_model(spec.id, model)
    route = dict(
        provider=spec.id, model=model, keys=keys, options=options,
        deadline=started + config.SCAN_DEADLINE_S,
    )

    paragraphs = paragraph_sentences(text)
    sentences = [s for p in paragraphs for s in p]
    if not sentences:
        raise DetectorError("Paste a few full sentences to rewrite.", status=400)

    # 1. Labels: reuse the client's last check; label only what's missing.
    #    Client labels are trusted for this request but never cached.
    known = {
        normalize_text(str(item.get("text", ""))): str(item.get("classification", "")).upper()
        for item in (labels or [])
        if isinstance(item, dict)
    }
    infos: list[dict] = []
    missing = [s for s in dict.fromkeys(sentences) if s not in known]
    if missing and (only_flagged or score_after_rewrite):
        found_labels, info = label_sentences(missing, **route)
        known.update(found_labels)
        infos.append(info)
    before_labels = [known.get(s, "HUMAN") for s in sentences]
    targets = {i for i, label in enumerate(before_labels) if label in FLAGGED or not only_flagged}
    score_before = _score(before_labels) if all(s in known for s in sentences) else None
    style_before = style.analyze_style(text)

    def finish(new_text: str, html: str, applied: list[Passage], rewrites: dict[int, str],
               reused: int, warnings: list[str], score_after: int | None) -> dict:
        models_used = next((i["models_used"] for i in reversed(infos) if i.get("models_used")), [])
        notes = []
        if any(PROVIDERS[m["provider"]].watermarks_output for m in models_used if m["provider"] in PROVIDERS):
            notes.append(
                "Rewritten with Google Gemini, whose output may carry Google's invisible SynthID "
                "watermark. Your own edits weaken it."
            )
        return {
            "text": new_text,
            "html": html,
            # The authorship trail: what was yours, what the model rewrote, and
            # which details you supplied.
            "changes": [
                {"before": p.text, "after": rewrites[p.number], "facts": p.facts} for p in applied
            ],
            "notes": notes,
            "passages": len(applied),
            "rewritten": sum(len(p.indices) for p in applied),
            "targets": len(targets),
            "reused": reused,
            "score_before": score_before,
            "score_after": score_after,
            "style": {
                "before": style_before.as_dict(),
                "after": style.analyze_style(new_text).as_dict(),
                "human_rhythm": list(style.HUMAN_CV),
            },
            "tips": style.grounding_tips(new_text),
            "tone": tone,
            "voice": voice,
            "usage": _merge_usage(infos),
            "warnings": warnings,
            "fallbacks": [s for i in infos for s in i.get("fallbacks", [])],
            "models_used": models_used,
            "elapsed_ms": int((time.monotonic() - started) * 1000),
        }

    if not targets:
        return finish(text, replace_spans(text, [])[1], [], {}, 0, [], score_before)

    # 2. Rewrite flagged passages (cache, validation, one targeted retry).
    passages = _passages(paragraphs, targets, whole_paragraphs=not only_flagged, facts=facts)
    rewrites, run_infos, reused, warnings = _rewrite(
        passages, tone=tone, voice=voice, route=route, fresh=fresh
    )
    infos += run_infos

    spans = locate_sentences(text, sentences)

    def apply() -> tuple[str, str, list[Passage]]:
        edits, applied = [], []
        for passage in passages:
            if passage.number not in rewrites:
                continue
            first, last = spans[passage.indices[0]], spans[passage.indices[-1]]
            if first is None or last is None:
                continue  # couldn't place it safely; leave the original
            edits.append((first[0], last[1], rewrites[passage.number]))
            applied.append(passage)
        new_text, html = replace_spans(text, edits)
        return new_text, html, applied

    new_text, html, applied = apply()
    skipped = len([p for p in passages if p.number in rewrites]) - len(applied)
    if skipped:
        warnings.append(f"{skipped} passage(s) couldn't be placed back safely and were left as written.")

    # 3. Re-score: unchanged sentences keep their labels; only new ones are sent.
    def rescore(candidate_text: str) -> int:
        new_sentences = document_sentences(candidate_text)
        unseen = [s for s in dict.fromkeys(new_sentences) if s not in known]
        if unseen:
            found, info = label_sentences(unseen, **route)
            known.update(found)
            infos.append(info)
        return _score([known.get(s, "HUMAN") for s in new_sentences])

    score_after = rescore(new_text) if (score_after_rewrite or second_pass) else None

    # 4. Extra polish: passages whose rewrite still reads generic get two new
    #    versions built differently (one request); the better one is kept.
    if second_pass:
        still = [
            p for p in applied
            if any(known.get(s) in FLAGGED for s in document_sentences(rewrites[p.number]))
        ]
        if still:
            notes = {
                p.number: [
                    f"Your last version still reads generic, with the same shape as before: "
                    f"“{rewrites[p.number]}”. Rebuild it rather than rewording it: different "
                    "sentence count, uneven lengths, plainer verbs, a clear stance."
                ]
                for p in still
            }
            again, more_infos, _, more = _rewrite(
                still, tone=tone, voice=voice, route=route, fresh=True, extra_notes=notes, variants=2
            )
            infos += more_infos
            warnings += [w for w in more if w not in warnings]
            rewrites.update(again)
            new_text, html, applied = apply()
            score_after = rescore(new_text)

    unchanged = len(targets) - sum(len(p.indices) for p in applied)
    if unchanged > 0 and only_flagged:
        warnings.append(f"{unchanged} flagged sentence(s) were left as they were.")
    return finish(new_text, html, applied, rewrites, reused, warnings, score_after)
