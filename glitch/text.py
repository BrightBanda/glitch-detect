"""Sentence splitting, chunking, and highlight reconstruction."""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Iterable

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(\[])')
_PARA_SPLIT_RE = re.compile(r"\n\s*\n")
WS_RE = re.compile(r"\s+")

LABELS = ("AI", "REFINED", "HUMAN")
CSS_CLASS = {"AI": "ai", "REFINED": "refined", "HUMAN": "human"}


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
