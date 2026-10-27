"""Turn a free model's (often messy) reply into classified sentences.

Degrades in four stages: strict JSON → trailing-comma repair → structural
regex (survives truncation and prose wrappers) → ParseError with a snippet.
"""

from __future__ import annotations

import json
import re

from .errors import ParseError
from .text import WS_RE, Sentence

LABEL_MAP = {
    "AI": "AI",
    "AI-GENERATED": "AI",
    "AI GENERATED": "AI",
    "MACHINE": "AI",
    "MACHINE-GENERATED": "AI",
    "MACHINE GENERATED": "AI",
    "GENERATED": "AI",
    "LLM": "AI",
    "BOT": "AI",
    "SYNTHETIC": "AI",
    "REFINED": "REFINED",
    "AI-REFINED": "REFINED",
    "AI REFINED": "REFINED",
    "PARAPHRASED": "REFINED",
    "AI-PARAPHRASED": "REFINED",
    "AI PARAPHRASED": "REFINED",
    "REPHRASED": "REFINED",
    "EDITED": "REFINED",
    "AI-EDITED": "REFINED",
    "POLISHED": "REFINED",
    "MIXED": "REFINED",
    "HYBRID": "REFINED",
    "AI-ASSISTED": "REFINED",
    "AI ASSISTED": "REFINED",
    "HUMAN": "HUMAN",
    "HUMAN-WRITTEN": "HUMAN",
    "HUMAN WRITTEN": "HUMAN",
    "AUTHENTIC": "HUMAN",
    "ORIGINAL": "HUMAN",
    "NATURAL": "HUMAN",
}

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")
_OBJ_FORWARD_RE = re.compile(
    r'\{\s*"text"\s*:\s*"(?P<text>(?:\\.|[^"\\])*)"\s*,\s*'
    r'"classification"\s*:\s*"(?P<label>[^"\\]*)"\s*\}',
    re.DOTALL,
)
_OBJ_REVERSE_RE = re.compile(
    r'\{\s*"classification"\s*:\s*"(?P<label>[^"\\]*)"\s*,\s*'
    r'"text"\s*:\s*"(?P<text>(?:\\.|[^"\\])*)"\s*\}',
    re.DOTALL,
)


def content_to_text(content) -> str:
    """Normalise OpenAI-style message content (str | list of parts | None)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for part in content:
            if isinstance(part, str):
                out.append(part)
            elif isinstance(part, dict):
                if isinstance(part.get("text"), str):
                    out.append(part["text"])
                elif isinstance(part.get("content"), str):
                    out.append(part["content"])
        return "".join(out)
    return ""


def strip_code_fences(raw: str) -> str:
    """Remove ```json ... ``` wrappers, keeping the largest fenced block."""
    if not raw:
        return ""
    cleaned = raw.strip().lstrip("\ufeff")
    matches = _FENCE_RE.findall(cleaned)
    if matches:
        return max(matches, key=len).strip()
    # An unterminated fence (truncated response) still needs handling.
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json|JSON)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def extract_outermost_json(raw: str) -> str:
    """Return the substring from the first '{' to the last '}'."""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1:
        # Maybe the model returned a bare array.
        start, end = raw.find("["), raw.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return raw
    return raw[start : end + 1]


def _json_unescape(value: str) -> str:
    try:
        return json.loads(f'"{value}"')
    except Exception:
        return value


def _items_from_payload(payload) -> list:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in ("sentences", "sentence", "results", "data", "analysis"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                for nested in ("sentences", "results"):
                    if isinstance(value.get(nested), list):
                        return value[nested]
    return []


def _item_fields(item) -> tuple[str, str]:
    if not isinstance(item, dict):
        return "", ""
    text = ""
    for key in ("text", "sentence", "content", "quote"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            text = value
            break
    label = ""
    for key in ("classification", "label", "type", "class", "category"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            label = value
            break
    return text, label


def normalize_label(label: str) -> str:
    key = (label or "").strip().upper().replace("_", "-")
    key = WS_RE.sub(" ", key)
    return LABEL_MAP.get(key, "")


_LABEL_TOKEN_RE = re.compile(
    r'"\s*(' + "|".join(re.escape(k) for k in sorted(LABEL_MAP, key=len, reverse=True)) + r')\s*"',
    re.IGNORECASE,
)
# Up to this share of missing labels is padded (as HUMAN) instead of failing.
_MAX_MISSING_SHARE = 0.1


def _labels_from_payload(payload) -> list:
    """Accept {"labels": [...]}, a bare list, sentence objects, or {"1": "AI"}."""
    if isinstance(payload, dict):
        for key in ("labels", "classifications", "results", "sentences"):
            if isinstance(payload.get(key), list):
                payload = payload[key]
                break
        else:
            numbered = [(k, v) for k, v in payload.items() if str(k).strip().isdigit()]
            if numbered:
                return [v for _, v in sorted(numbered, key=lambda kv: int(kv[0]))]
            return []
    if not isinstance(payload, list):
        return []
    labels = []
    for item in payload:
        if isinstance(item, dict):
            item = _item_fields(item)[1]
        labels.append(item if isinstance(item, str) else "")
    return labels


def parse_labels(raw: str, expected: int) -> tuple[list[str], list[str]]:
    """Parse a compact reply into exactly ``expected`` canonical labels."""
    warnings: list[str] = []
    if not raw or not raw.strip():
        raise ParseError("The model returned an empty response.")

    candidate = extract_outermost_json(strip_code_fences(raw))
    payload = None
    for text in (candidate, _TRAILING_COMMA_RE.sub(r"\1", candidate)):
        try:
            payload = json.loads(text)
            break
        except Exception:
            continue

    labels = _labels_from_payload(payload) if payload is not None else []
    if not labels:
        # Truncated or prose-wrapped: take quoted label words in order.
        labels = [m.group(1) for m in _LABEL_TOKEN_RE.finditer(candidate)]
        if labels:
            warnings.append("Recovered labels with the fallback parser.")

    normalized = [normalize_label(label) for label in labels]
    unknown = sum(1 for label in normalized if not label)
    normalized = [label or "HUMAN" for label in normalized]
    if unknown:
        warnings.append(
            f"{unknown} sentence(s) had an unrecognised label and were treated as HUMAN."
        )

    if len(normalized) > expected:
        normalized = normalized[:expected]
        warnings.append("The model returned extra labels; they were ignored.")
    missing = expected - len(normalized)
    if missing > max(1, int(expected * _MAX_MISSING_SHARE)) or not normalized:
        snippet = WS_RE.sub(" ", raw)[:280]
        raise ParseError(
            f"The model labelled {len(normalized)} of {expected} sentences.",
            detail=f"Raw response began: {snippet}",
        )
    if missing:
        normalized += ["HUMAN"] * missing
        warnings.append(f"{missing} sentence(s) were left unlabelled and treated as HUMAN.")
    return normalized, warnings


def parse_analysis(raw: str) -> tuple[list[Sentence], list[str]]:
    """Parse a model response into sentences. Raises ParseError if impossible."""
    warnings: list[str] = []
    if not raw or not raw.strip():
        raise ParseError("The model returned an empty response.")

    candidate = extract_outermost_json(strip_code_fences(raw))

    payload = None
    try:
        payload = json.loads(candidate)
    except Exception:
        repaired = _TRAILING_COMMA_RE.sub(r"\1", candidate)
        try:
            payload = json.loads(repaired)
            warnings.append("Repaired malformed JSON (trailing commas).")
        except Exception:
            payload = None

    items = _items_from_payload(payload) if payload is not None else []

    if not items:
        for regex in (_OBJ_FORWARD_RE, _OBJ_REVERSE_RE):
            items = [
                {
                    "text": _json_unescape(match.group("text")),
                    "classification": match.group("label"),
                }
                for match in regex.finditer(candidate)
            ]
            if items:
                warnings.append("Recovered sentences with the fallback regex parser.")
                break

    sentences: list[Sentence] = []
    unknown = 0
    for item in items:
        text, label = _item_fields(item)
        text = WS_RE.sub(" ", text or "").strip()
        if not text:
            continue
        classification = normalize_label(label)
        if not classification:
            classification = "HUMAN"
            unknown += 1
        sentences.append(Sentence(text=text, classification=classification))

    if unknown:
        warnings.append(
            f"{unknown} sentence(s) had an unrecognised label and were treated as HUMAN."
        )

    if not sentences:
        snippet = WS_RE.sub(" ", raw)[:280]
        raise ParseError(
            "Could not extract any sentences from the model response.",
            detail=f"Raw response began: {snippet}",
        )

    return sentences, warnings


_REWRITE_PAIR_RE = re.compile(r'"(\d+)"\s*:\s*"((?:\\.|[^"\\])*)"', re.DOTALL)


def parse_rewrites(raw: str, expected: list[int], variants: int = 1) -> tuple[dict, list[str]]:
    """Parse ``{"rewrites": {"3": "..."}}`` (or close variants) into {number: text}.

    With ``variants`` > 1 the model sends ``{"3": ["A", "B"]}`` and the result
    maps each number to a list of versions (a single string still counts).
    """
    if not raw or not raw.strip():
        raise ParseError("The model returned an empty response.")
    candidate = extract_outermost_json(strip_code_fences(raw))
    payload = None
    for text in (candidate, _TRAILING_COMMA_RE.sub(r"\1", candidate)):
        try:
            payload = json.loads(text)
            break
        except Exception:
            continue

    found: dict[int, str | list] = {}
    if isinstance(payload, dict):
        inner = payload.get("rewrites", payload)
        if isinstance(inner, dict):
            for k, v in inner.items():
                if str(k).strip().isdigit() and (isinstance(v, str) or isinstance(v, list)):
                    found[int(k)] = v
        elif isinstance(inner, list):
            payload = inner
    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            number = next((item[k] for k in ("n", "id", "number", "index") if k in item), None)
            text = next((item[k] for k in ("text", "rewrite", "rewritten") if k in item), None)
            if str(number).strip().isdigit() and isinstance(text, str):
                found[int(number)] = text
    if not found:
        found = {int(n): _json_unescape(t) for n, t in _REWRITE_PAIR_RE.findall(candidate)}

    wanted = set(expected)
    versions: dict[int, list[str]] = {}
    for n, value in found.items():
        texts = value if isinstance(value, list) else [value]
        texts = [WS_RE.sub(" ", t).strip() for t in texts if isinstance(t, str)]
        if n in wanted and any(texts):
            versions[n] = [t for t in texts if t][: max(1, variants)]
    rewrites = versions if variants > 1 else {n: v[0] for n, v in versions.items()}
    if not rewrites:
        snippet = WS_RE.sub(" ", raw)[:280]
        raise ParseError(
            "The model didn't return any rewritten sentences.",
            detail=f"Raw response began: {snippet}",
        )
    warnings = []
    missing = len(wanted) - len(rewrites)
    if missing:
        warnings.append(f"{missing} sentence(s) were left unchanged.")
    return rewrites, warnings
