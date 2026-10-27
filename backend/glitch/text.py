"""Sentence splitting, chunking, and highlight reconstruction."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(\[])')
_PARA_SPLIT_RE = re.compile(r"\n\s*\n")
WS_RE = re.compile(r"\s+")

LABELS = ("AI", "REFINED", "HUMAN")
CSS_CLASS = {"AI": "ai", "REFINED": "refined", "HUMAN": "human"}


# Lookalike letters from other scripts, used to dodge detectors or by accident
# (pasted from PDFs). Mapped only inside otherwise-Latin words.
_CONFUSABLES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x", "і": "i", "ј": "j",
    "ѕ": "s", "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O", "Р": "P",
    "С": "C", "Т": "T", "Х": "X", "ο": "o", "Ο": "O", "α": "a", "ν": "v",
})
_MIXED_WORD_RE = re.compile(r"\w*[A-Za-z]\w*")
_DASHES = str.maketrans({"\u2010": "-", "\u2011": "-", "\u00ad": ""})


def normalize_text(text: str) -> str:
    """Clean pasted text the way detectors do (reports/How AI humanizers work.md).

    NFKC (fullwidth letters, ligatures), drop invisible format characters
    (zero-width spaces/joiners, BOM, soft hyphens), plain hyphens, and map
    lookalike Cyrillic/Greek letters inside Latin words. Line breaks and
    ordinary punctuation are kept, so paragraphs survive.
    """
    text = unicodedata.normalize("NFKC", text or "").translate(_DASHES)
    text = "".join(
        ch for ch in text if ch in "\n\t" or unicodedata.category(ch) not in ("Cf", "Cc")
    )
    return _MIXED_WORD_RE.sub(lambda m: m.group(0).translate(_CONFUSABLES), text)


def has_hidden_characters(text: str) -> bool:
    """Invisible format characters or lookalike letters (a red flag to detectors)."""
    if any(unicodedata.category(ch) == "Cf" for ch in text):
        return True
    return any(m.group(0) != m.group(0).translate(_CONFUSABLES) for m in _MIXED_WORD_RE.finditer(text))


@dataclass
class Sentence:
    text: str
    classification: str


def split_sentences(text: str) -> list[str]:
    """Best-effort sentence splitter used for chunking and coverage estimates."""
    cleaned = WS_RE.sub(" ", text).strip()
    if not cleaned:
        return []
    return [p.strip() for p in _SENT_SPLIT_RE.split(cleaned) if p.strip()]


def paragraph_sentences(text: str) -> list[list[str]]:
    """Sentences grouped by paragraph."""
    return [found for para in _PARA_SPLIT_RE.split(text) if (found := split_sentences(para))]


def document_sentences(text: str) -> list[str]:
    """All sentences in order, split per paragraph so headings stand alone."""
    return [s for para in paragraph_sentences(text) for s in para]


def group_sentences(sentences: list[str], max_words: int = 900) -> list[list[str]]:
    """Pack sentences into request-sized groups, preserving order."""
    max_words = max(120, min(int(max_words), 2000))
    groups: list[list[str]] = []
    words = 0
    for sentence in sentences:
        count = len(sentence.split())
        if groups and words + count <= max_words:
            groups[-1].append(sentence)
            words += count
        else:
            groups.append([sentence])
            words = count
    return groups


def chunk_text(text: str, max_words: int = 900) -> list[str]:
    """Split on paragraph boundaries, then sentence boundaries if needed."""
    max_words = max(120, min(int(max_words), 2000))
    text = text.strip()
    if len(text.split()) <= max_words:
        return [text]

    chunks: list[str] = []
    buf: list[str] = []
    buf_words = 0

    def flush() -> None:
        nonlocal buf, buf_words
        if buf:
            chunks.append("\n\n".join(buf).strip())
            buf, buf_words = [], 0

    for para in _PARA_SPLIT_RE.split(text):
        para = para.strip()
        if not para:
            continue
        words = len(para.split())
        if words > max_words:
            flush()
            cur: list[str] = []
            cur_words = 0
            for sentence in split_sentences(para):
                sw = len(sentence.split())
                if cur and cur_words + sw > max_words:
                    chunks.append(" ".join(cur).strip())
                    cur, cur_words = [sentence], sw
                else:
                    cur.append(sentence)
                    cur_words += sw
            if cur:
                chunks.append(" ".join(cur).strip())
            continue

        if buf and buf_words + words > max_words:
            flush()
        buf.append(para)
        buf_words += words

    flush()
    return [c for c in chunks if c]


def _find_sentence(raw: str, sentence: str, start: int):
    tokens = sentence.split()
    if not tokens:
        return None
    pattern = r"\s+".join(re.escape(t) for t in tokens)
    return re.compile(pattern, re.IGNORECASE).search(raw, start)


def build_highlight_html(sentences: Iterable[Sentence], source_text: str) -> str:
    """Rebuild the document as escaped <span>s, preserving paragraph breaks.

    When a returned sentence can be located in the source, the source wording
    is shown (models sometimes normalise quotes or casing); otherwise the
    model's text is used as-is.
    """
    paragraphs: list[list[tuple[str, str]]] = [[]]
    cursor = 0
    raw = source_text or ""

    for sentence in sentences:
        shown = sentence.text
        match = _find_sentence(raw, sentence.text, cursor) if raw else None
        if match is not None:
            if raw[cursor : match.start()].count("\n") >= 2 and paragraphs[-1]:
                paragraphs.append([])
            shown = WS_RE.sub(" ", match.group(0))
            cursor = match.end()
        paragraphs[-1].append((shown, sentence.classification))

    blocks: list[str] = []
    for group in paragraphs:
        if not group:
            continue
        spans = [
            f'<span class="hl {CSS_CLASS[label]}" data-label="{label}" '
            f'title="{label} — {len(text.split())} words">{html.escape(text)}</span>'
            for text, label in group
        ]
        blocks.append('<p class="doc-para">' + " ".join(spans) + "</p>")

    if not blocks:
        return '<p class="doc-para"><span class="hl human">No sentences returned.</span></p>'
    return "\n".join(blocks)


def locate_sentences(source: str, sentences: list[str]) -> list[tuple[int, int] | None]:
    """Character span of each sentence in ``source``, searching in order."""
    spans: list[tuple[int, int] | None] = []
    cursor = 0
    for sentence in sentences:
        match = _find_sentence(source, sentence, cursor)
        if match is None:
            spans.append(None)
            continue
        spans.append((match.start(), match.end()))
        cursor = match.end()
    return spans


def _paragraph_html(segments: list[tuple[str, bool]]) -> str:
    """Escaped HTML with changed segments marked, paragraphs preserved."""
    out = []
    for text, changed in segments:
        escaped = html.escape(text)
        out.append(f'<span class="hw">{escaped}</span>' if changed else escaped)
    body = "".join(out).strip()
    paragraphs = [p.replace("\n", "<br>") for p in _PARA_SPLIT_RE.split(body) if p.strip()]
    return "\n".join(f'<p class="doc-para">{p}</p>' for p in paragraphs)


def replace_spans(source: str, edits: list[tuple[int, int, str]]) -> tuple[str, str]:
    """Apply non-overlapping ``(start, end, new_text)`` edits to ``source``.

    Returns ``(new text, HTML with replaced parts marked)``. Everything between
    edits — spacing, line and paragraph breaks — is kept exactly.
    """
    segments: list[tuple[str, bool]] = []
    cursor = 0
    for start, end, new in sorted(edits):
        segments.append((source[cursor:start], False))
        segments.append((new, True))
        cursor = end
    segments.append((source[cursor:], False))
    return "".join(t for t, _ in segments), _paragraph_html(segments)


def replace_sentences(
    source: str, sentences: list[str], replacements: dict[int, str]
) -> tuple[str, str]:
    """Swap ``replacements`` (sentence index -> new text) into ``source``."""
    spans = locate_sentences(source, sentences)
    edits = [
        (span[0], span[1], replacements[i])
        for i, span in enumerate(spans)
        if span is not None and i in replacements
    ]
    return replace_spans(source, edits)
