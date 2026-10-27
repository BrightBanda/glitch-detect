"""Before/after measurement for the humanizer (reports/How AI humanizers work.md, Phase 0).

Runs ``humanize.humanize()`` against real providers on a fixed corpus of
AI-written passages and writes one JSONL record per run, then prints medians
with bootstrap 95% intervals. Compare two arms (e.g. before/after a change):

    python -m eval.humanize_eval corpus            # generate the AI corpus once
    python -m eval.humanize_eval run --label v3 --runs 2
    python -m eval.humanize_eval compare v3 v4

Metrics are deliberately separate and honest:

* ``score_after`` — the app's own detector (a *proxy*, not an external tool).
* ``lost`` / ``invented`` — names and numbers dropped from / added to the source.
* ``claims`` — first-person experience claims ("in my experience") not in the source.
* ``overlap`` — content-word overlap with the source (crude meaning check).
* ``rhythm`` / ``phrases`` — sentence-length CV and AI-phrase count after.
* ``tokens_per_150w`` and ``elapsed_s`` — cost.

Run from ``backend/``. Uses the server's keys (``.env`` / ``instance/keys.json``).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

from glitch import analyzer, client, keys, style
from glitch.humanize import REWRITE_CACHE, humanize
from glitch.prompt import HUMANIZE_VERSION
from glitch.providers import get_provider
from glitch.routing import ScanOptions
from glitch.text import document_sentences, normalize_text

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus.jsonl"
RESULTS = HERE / "results"

TOPICS = [
    ("essay", "why remote work changed how young professionals build careers"),
    ("essay", "the role of social media in modern political debate"),
    ("report", "the impact of solar power adoption on small businesses"),
    ("report", "how a mid-sized city could reduce traffic congestion"),
    ("blog", "tips for staying productive while studying at home"),
    ("blog", "what beginners should know before starting to run"),
    ("essay", "whether standardized testing measures real learning"),
    ("report", "the benefits and risks of mobile money for rural households"),
]
_CONTENT_RE = re.compile(r"[a-z]{4,}")


def generate_corpus(provider: str) -> None:
    spec = get_provider(provider)
    key = keys.resolve_request_keys()[spec.id]
    rows = []
    for register, topic in TOPICS:
        raw = client.request_chunk(
            [{"role": "user", "content": (
                f"Write a {register} paragraph (about 150 words, one or two paragraphs, plain text, "
                f"no title or markdown) about {topic}."
            )}],
            key, spec.default_model, spec.id, max_tokens=1200, temperature=0.7,
        )
        rows.append({"id": len(rows) + 1, "register": register, "text": raw.strip()})
        print(f"  {rows[-1]['id']}: {len(raw.split())} words", file=sys.stderr)
    CORPUS.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {CORPUS}")


def _content(text: str) -> set[str]:
    return set(_CONTENT_RE.findall(text.lower()))


def measure(before: str, after: str) -> dict:
    """Same fact rules as the app (style.check_rewrite / style.invented)."""
    before = normalize_text(before)
    facts, _ = style.check_rewrite(before, after)
    content_b, content_a = _content(before), _content(after)
    report = style.analyze_style(after)
    return {
        "lost": [f for f in facts if f.startswith("Keep")],
        "invented": [f for f in facts if "add" in f],
        "claims": [f for f in facts if "experiences" in f],
        "overlap": round(len(content_a & content_b) / max(1, len(content_b)), 3),
        "rhythm": report.rhythm,
        "phrases": report.phrase_count,
    }


def remeasure(label: str) -> None:
    """Recompute metrics for saved runs (after changing the rules)."""
    corpus = {r["id"]: r["text"] for r in map(json.loads, CORPUS.read_text().splitlines())}
    path = RESULTS / f"{label}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        if "text" in row:
            row.update(measure(corpus[row["id"]], row["text"]))
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def run(label: str, runs: int, provider: str, limit: int) -> None:
    rows = [json.loads(line) for line in CORPUS.read_text().splitlines()][:limit]
    spec = get_provider(provider)
    request_keys = keys.resolve_request_keys()
    options = ScanOptions()
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"{label}.jsonl"
    with out.open("a") as fh:
        for row in rows:
            # Label once so every run starts from the same detector verdict.
            sentences = document_sentences(row["text"])
            found, _ = analyzer.label_sentences(
                sentences, provider=spec.id, model=spec.default_model, keys=request_keys,
                options=options, deadline=time.monotonic() + 200,
            )
            labels = [{"text": s, "classification": found.get(s, "HUMAN")} for s in sentences]
            for attempt in range(runs):
                REWRITE_CACHE.clear()
                started = time.monotonic()
                record = {"label": label, "version": HUMANIZE_VERSION, "id": row["id"],
                          "register": row["register"], "run": attempt + 1, "provider": spec.id}
                try:
                    result = humanize(row["text"], provider=spec.id, keys=request_keys,
                                      labels=labels, fresh=True, options=options)
                except Exception as exc:  # noqa: BLE001 - record and keep going
                    record.update(error=str(getattr(exc, "message", exc))[:200],
                                  detail=str(getattr(exc, "detail", ""))[:300])
                    print(f"  #{row['id']} run {attempt + 1}: ERROR {record['error']}", file=sys.stderr)
                    fh.write(json.dumps(record) + "\n")
                    continue
                words = len(row["text"].split())
                record.update(
                    score_before=result["score_before"], score_after=result["score_after"],
                    tokens=result["usage"]["total_tokens"],
                    tokens_per_150w=round(result["usage"]["total_tokens"] * 150 / max(1, words)),
                    elapsed_s=round(time.monotonic() - started, 1),
                    models=result["models_used"], text=result["text"],
                    warnings=result["warnings"], fallbacks=len(result["fallbacks"]),
                    **measure(row["text"], result["text"]),
                )
                fh.write(json.dumps(record) + "\n")
                fh.flush()
                print(f"  #{row['id']} run {attempt + 1}: {record['score_before']}% -> "
                      f"{record['score_after']}%  {record['tokens']} tok  {record['elapsed_s']}s",
                      file=sys.stderr)
    print(f"wrote {out}")


def _ci(values: list[float], reps: int = 2000) -> tuple[float, float, float]:
    rng = random.Random(7)
    medians = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(reps))
    return statistics.median(values), medians[int(0.025 * reps)], medians[int(0.975 * reps)]


def summarize(label: str) -> dict:
    rows = [json.loads(line) for line in (RESULTS / f"{label}.jsonl").read_text().splitlines()]
    ok = [r for r in rows if "error" not in r]
    summary = {"label": label, "runs": len(rows), "errors": len(rows) - len(ok)}
    if not ok:
        return summary
    for name in ("score_before", "score_after", "tokens_per_150w", "elapsed_s", "overlap", "rhythm", "phrases"):
        values = [r[name] for r in ok if r.get(name) is not None]
        if values:
            summary[name] = tuple(round(v, 2) for v in _ci(values))
    summary["runs_with_lost_facts"] = sum(1 for r in ok if r["lost"])
    summary["runs_with_invented_facts"] = sum(1 for r in ok if r["invented"])
    summary["runs_with_invented_claims"] = sum(1 for r in ok if r["claims"])
    summary["under_50pct"] = sum(1 for r in ok if (r["score_after"] or 0) < 50)
    return summary


def compare(labels: list[str]) -> None:
    for label in labels:
        s = summarize(label)
        print(f"\n== {label}  ({s['runs']} runs, {s['errors']} errors)")
        for key, value in s.items():
            if key in ("label", "runs", "errors"):
                continue
            if isinstance(value, tuple):
                print(f"  {key:26} median {value[0]:>8}   95% CI [{value[1]}, {value[2]}]")
            else:
                print(f"  {key:26} {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    gen = sub.add_parser("corpus")
    gen.add_argument("--provider", default="groq")
    go = sub.add_parser("run")
    go.add_argument("--label", required=True)
    go.add_argument("--runs", type=int, default=2)
    go.add_argument("--limit", type=int, default=len(TOPICS))
    go.add_argument("--provider", default="groq")
    again = sub.add_parser("remeasure")
    again.add_argument("labels", nargs="+")
    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("labels", nargs="+")
    args = parser.parse_args()
    if args.cmd == "corpus":
        generate_corpus(args.provider)
    elif args.cmd == "run":
        run(args.label, args.runs, args.provider, args.limit)
    elif args.cmd == "remeasure":
        for label in args.labels:
            remeasure(label)
    else:
        compare(args.labels)


if __name__ == "__main__":
    main()
