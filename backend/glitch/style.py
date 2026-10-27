"""Local, zero-token style analysis grounded in how detectors actually work.

Detectors weigh three tiers of evidence (see mechanics-of-ai-writing-detection.md
§2.1, §2.8). Word-level tells (T1) are the weakest; rhythm and discourse
structure (T2) and grounding (T3) carry most of the signal and survive simple
paraphrasing. This module measures the T1/T2 features a rewrite can fix, and
spots the T3 gaps only the writer can fill:

* sentence-length dispersion — CV = σ/μ; human prose ≈ 0.5–0.7, default LLM
  output ≈ 0.3–0.45 (§2.3.1);
* the LLM lexicon (§2.2.3) and rhetorical templates: tricolons, antithesis,
  "not only … but also", copula avoidance, stakes openers, recap closers,
  trailing participial clauses (§2.3.3, Reinhart et al.);
* sentence-initial connectives and uniform hedging (§2.3.2, §2.5.3);
* unanchored paragraphs — no numbers, names, or particulars (§2.5).

It feeds the humanizer three ways: per-passage rewrite instructions, a free
pass/fail check on each rewrite, and before/after metrics for the UI.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from .text import WS_RE, split_sentences

# §2.2.3 plus closely related high-frequency items.
LEXICON = (
    "delve", "delves", "delving", "underscore", "underscores", "underscoring",
    "showcase", "showcases", "showcasing", "crucial", "pivotal", "vital",
    "testament", "tapestry", "landscape", "realm", "navigate", "navigating",
    "foster", "fosters", "fostering", "leverage", "leverages", "leveraging",
    "robust", "seamless", "seamlessly", "holistic", "multifaceted",
    "transformative", "unprecedented", "streamline", "streamlines", "empower",
    "empowers", "empowering", "synergy", "paradigm", "intricate", "embark",
    "elevate", "unlock", "unlocks", "harness", "harnessing", "cutting-edge",
    "ever-evolving", "game-changer", "stakeholders", "plethora", "myriad",
    "meticulous", "meticulously", "noteworthy", "commendable", "invaluable",
    "bustling", "vibrant", "enhance", "enhances", "enhancing",
)
_LEXICON_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in LEXICON) + r")\b", re.IGNORECASE)

# (pattern, plain-language description used in rewrite instructions)
TEMPLATES = (
    (r"\bnot only\b.{1,80}?\bbut also\b", "“not only … but also”"),
    (r"\bit(?:'s| is) (?:important|worth|crucial|essential) to (?:note|mention|remember|recognize)",
     "“it is important to note”"),
    (r"\b(?:serves|serve|stands|stand) as\b", "“serves as / stands as” (just say “is”)"),
    (r"\bplays? an? (?:crucial|key|vital|pivotal|important|significant|central) role\b",
     "“plays a crucial role”"),
    (r"(?:^|(?<=[.!?] ))(?:in today's|in an era of|in a world where|in the ever|in this day and age)",
     "a scene-setting opener"),
    (r"\bnot (?:merely|just|simply|only) [^.;]{1,60}[;,—–-] (?:it|but)\b", "a “not just X, but Y” antithesis"),
    (r"(?:^|(?<=[.!?] ))(?:ultimately|in conclusion|in summary|overall|all in all|in essence)\b",
     "a summing-up opener"),
    (r"\bwhile [^.]{1,80}, (?:it|they|this) (?:also )?(?:raises?|presents?|poses?|comes? with)\b",
     "a both-sides closing line"),
    (r", (?:ensuring|highlighting|underscoring|showcasing|fostering|enabling|allowing|"
     r"demonstrating|reflecting|paving|making it|leading to)\b", "a trailing “…, ensuring/highlighting” clause"),
)
_TEMPLATE_RES = [(re.compile(p, re.IGNORECASE), label) for p, label in TEMPLATES]

# "a, b, and c" with short items: the rhythm-completing tricolon (§2.3.3).
# A list item of 1-5 words. Each word is one unambiguous token, so matching
# stays linear (nested "\w+[\w'-]*" groups backtrack exponentially).
_ITEM = r"[\w'-]+(?: [\w'-]+){0,4}"
_TRICOLON_RE = re.compile(rf"\b({_ITEM}), ({_ITEM}),? (?:and|or) ({_ITEM})", re.IGNORECASE)
_CONNECTIVE_START_RE = re.compile(
    r"^(moreover|furthermore|additionally|consequently|notably|ultimately|overall|"
    r"in addition|as a result|thus|hence|importantly|indeed)\b",
    re.IGNORECASE,
)
_HEDGE_RE = re.compile(
    r"\b(may|might|could|can|often|typically|generally|potentially|arguably|likely|"
    r"in many cases|tends? to)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"\d[\d,.]*%?")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_GENERIC_RE = re.compile(
    r"\b(many|various|numerous|several|studies|experts|research(?:ers)?|organizations|"
    r"businesses|people|individuals|society|recently|today|nowadays|modern)\b",
    re.IGNORECASE,
)
_PARA_SPLIT_RE = re.compile(r"\n\s*\n")

HUMAN_CV = (0.5, 0.7)
FLAT_CV = 0.35    # below this, rhythm reads as machine-even
TARGET_CV = 0.5   # a rewrite of 3+ sentences should reach the human range


def sentence_cv(sentences: list[str]) -> float | None:
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 2:
        return None
    mean = statistics.mean(lengths)
    return round(statistics.pstdev(lengths) / mean, 3) if mean else None


def lexicon_hits(text: str) -> list[str]:
    seen: dict[str, None] = {}
    for match in _LEXICON_RE.finditer(text):
        seen.setdefault(match.group(1).lower(), None)
    return list(seen)


def template_hits(text: str) -> list[str]:
    return [label for regex, label in _TEMPLATE_RES if regex.search(text)]


_ABSTRACT_RE = re.compile(r"\w(?:tion|sion|ity|ness|ment|ance|ence|ism)s?\b", re.IGNORECASE)


def tricolons(text: str) -> list[str]:
    """Rhythm-completing three-part lists: phrase items or abstract nouns.

    Plain lists ("eggs, milk and bread") are ordinary human writing and are
    not reported.
    """
    found = []
    for match in _TRICOLON_RE.finditer(text):
        phrase_items = len(match.group(2).split()) >= 2
        abstract = any(_ABSTRACT_RE.search(item) for item in match.groups())
        if phrase_items or abstract:
            found.append(match.group(0))
    return found


def names_and_numbers(text: str) -> tuple[set[str], set[str]]:
    """Proper names / acronyms (not sentence-initial) and numbers in ``text``."""
    names: set[str] = set()
    for sentence in split_sentences(text):
        words = _WORD_RE.findall(sentence)
        for word in words[1:]:
            if word == "I":
                continue
            if word[0].isupper() and (any(c.islower() for c in word[1:]) or len(word) >= 2):
                names.add(word)
    numbers = {n.rstrip(".,") for n in _NUMBER_RE.findall(text)}
    return names, {n for n in numbers if n}


@dataclass
class StyleReport:
    sentences: int
    rhythm: float | None              # sentence-length CV
    ai_phrases: list[str] = field(default_factory=list)
    lists_of_three: list[str] = field(default_factory=list)
    connective_starts: int = 0
    hedges_per_sentence: float = 0.0

    @property
    def phrase_count(self) -> int:
        return len(self.ai_phrases) + len(self.lists_of_three) + self.connective_starts

    def as_dict(self) -> dict:
        return {
            "sentences": self.sentences,
            "rhythm": self.rhythm,
            "ai_phrases": self.phrase_count,
            "examples": (self.ai_phrases + self.lists_of_three)[:6],
        }


def analyze_style(text: str) -> StyleReport:
    sentences = [s for p in _PARA_SPLIT_RE.split(text) for s in split_sentences(p)]
    hedges = len(_HEDGE_RE.findall(text))
    return StyleReport(
        sentences=len(sentences),
        rhythm=sentence_cv(sentences),
        ai_phrases=lexicon_hits(text) + template_hits(text),
        lists_of_three=tricolons(text),
        connective_starts=sum(1 for s in sentences if _CONNECTIVE_START_RE.match(s)),
        hedges_per_sentence=round(hedges / len(sentences), 2) if sentences else 0.0,
    )


def rewrite_notes(passage: str, *, document_rhythm: float | None = None) -> list[str]:
    """Specific instructions for rewriting one passage, from what it actually contains."""
    sentences = split_sentences(passage)
    lengths = [len(s.split()) for s in sentences]
    notes: list[str] = []

    cv = sentence_cv(sentences)
    if cv is not None and cv < FLAT_CV:
        span = f"{min(lengths)}" if min(lengths) == max(lengths) else f"{min(lengths)}–{max(lengths)}"
        notes.append(
            f"Its sentences are all {span} words long. Vary them a lot: "
            "include one short sentence (under 8 words) and let another run long; merging or "
            "splitting sentences is fine."
        )
    elif len(sentences) == 1 and lengths and lengths[0] > 22:
        notes.append("It is one long sentence; split it or tighten it.")
    elif len(sentences) == 1 and document_rhythm is not None and document_rhythm < FLAT_CV:
        notes.append("Make its length clearly different from the sentences around it.")

    words = lexicon_hits(passage)
    if words:
        notes.append("Replace these words with plain ones: " + ", ".join(words) + ".")
    for label in template_hits(passage):
        notes.append(f"Remove {label}.")
    for triple in tricolons(passage)[:2]:
        notes.append(f"Break up the three-part list “{triple}”: keep only what matters.")
    if any(_CONNECTIVE_START_RE.match(s) for s in sentences):
        notes.append("Don't open sentences with linking words like Moreover, Additionally or Ultimately.")
    if sentences and len(_HEDGE_RE.findall(passage)) / len(sentences) > 0.8:
        notes.append("Hedge only where something is genuinely uncertain.")
    names, numbers = names_and_numbers(passage)
    if names or numbers:
        keep = sorted(numbers, key=lambda n: passage.find(n)) + sorted(names, key=lambda n: passage.find(n))
        notes.append("Keep exactly: " + ", ".join(keep) + ".")
    return notes


_FIRST_PERSON_RE = re.compile(r"\b(I|I'm|I've|I'd|me|my|we|we're|our|us)\b")
_CONTRACTION_RE = re.compile(r"\b\w+'(?:s|t|re|ve|ll|d|m)\b", re.IGNORECASE)
_ASIDE_RE = re.compile(r"\(|—|–| - |;|:")


def voice_markers(text: str) -> set[str]:
    """Signs of a particular writer's voice (§2.5: stance, register, rhythm).

    Measured on live detectors, a passage with several of these — and no new
    facts — scored far lower than a clean but voiceless paraphrase.
    """
    sentences = split_sentences(text)
    found = set()
    if _FIRST_PERSON_RE.search(text):
        found.add("first person")
    if _CONTRACTION_RE.search(text.replace("’", "'")):
        found.add("contractions")
    if "?" in text:
        found.add("a question")
    if any(len(s.split()) <= 4 for s in sentences):
        found.add("a short fragment")
    if _ASIDE_RE.search(text):
        found.add("an aside")
    return found


# First-person *experience* claims (not opinions): "I think" is stance and is
# fine; "when I worked at…" is a made-up fact unless the writer supplied it.
_CLAIM_RE = re.compile(
    r"\b(?:I remember|I recall|in my (?:own )?experience|I(?:'ve| have) (?:seen|watched|worked|tried|met|lived)|"
    r"I once|I used to|when I (?:was|worked|tried|started|lived|visited|first|moved)|"
    r"(?:a|one) (?:friend|colleague|client|student) of mine|my (?:friend|colleague|boss|teacher|"
    r"mom|mum|dad|sister|brother|cousin|neighbou?r|manager|professor)|last (?:year|month|week|summer))\b",
    re.IGNORECASE,
)
_UNITS = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def _number_words(number: str) -> str | None:
    """'25' → 'twenty-five' (0–99), for matching numbers a rewrite spelled out."""
    if not number.isdigit() or int(number) > 99:
        return None
    n = int(number)
    if n < 20:
        return _UNITS[n]
    return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _UNITS[n % 10])


def _bare(number: str) -> str:
    return number.rstrip("%").rstrip(".,")


def _number_present(number: str, text: str) -> bool:
    """Is ``number`` in ``text`` as digits (ignoring a % sign) or spelled out?"""
    if re.search(rf"(?<![\d.,]){re.escape(_bare(number))}(?![\d])", text):
        return True
    words = _number_words(_bare(number))
    return bool(words) and re.search(rf"\b{words.replace('-', '[- ]')}\b", text, re.IGNORECASE) is not None


def _name_present(name: str, lowered: str) -> bool:
    """Case- and hyphen-insensitive ("Gen Z" vs "Gen-Z", "TikTok" vs "Tiktok")."""
    return name.lower() in lowered or name.lower() in lowered.replace("-", " ")


def invented(before: str, after: str, allowed: str = "") -> list[str]:
    """New particulars in a rewrite that neither the source nor the author supplied.

    ``allowed`` is author-supplied text (grounding answers): its names, numbers
    and experiences are fair game. Returns feedback lines (empty if clean).
    """
    source = f"{before}\n{allowed}"
    lowered = source.lower()
    names_src, numbers_src = names_and_numbers(source)
    names_new, numbers_new = names_and_numbers(after)
    numbers = sorted(n for n in numbers_new if not _number_present(n, source))
    names = sorted(n for n in names_new if not _name_present(n, lowered))
    claims = sorted({m.group(0) for m in _CLAIM_RE.finditer(after) if m.group(0).lower() not in lowered})
    problems = []
    if numbers:
        problems.append("Don't add numbers that aren't in the original: " + ", ".join(numbers) + ".")
    if names:
        problems.append("Don't add names that aren't in the original: " + ", ".join(names) + ".")
    if claims:
        problems.append(
            "Don't invent personal experiences (" + ", ".join(f"“{c}”" for c in claims)
            + "). Give an opinion instead, or keep it general."
        )
    return problems


def check_rewrite(
    before: str, after: str, *, voice: str = "faithful", allowed: str = ""
) -> tuple[list[str], list[str]]:
    """Free validation of a rewrite: (fact problems, remaining style problems).

    Fact problems — lost names/numbers, or invented ones — mean the rewrite
    can't be used as is; style problems only earn a retry.
    """
    names_before, numbers_before = names_and_numbers(before)
    lowered = after.lower()
    lost: list[str] = []
    missing_numbers = sorted(n for n in numbers_before if not _number_present(n, after))
    if missing_numbers:
        lost.append("Keep these numbers exactly: " + ", ".join(missing_numbers) + ".")
    missing_names = sorted(n for n in names_before if not _name_present(n, lowered))
    if missing_names:
        lost.append("Keep these names exactly: " + ", ".join(missing_names) + ".")
    lost += invented(before, after, allowed)

    problems: list[str] = []
    words = lexicon_hits(after)
    if words:
        problems.append("Still uses: " + ", ".join(words) + ". Replace them.")
    for label in template_hits(after):
        problems.append(f"Still has {label}.")
    for triple in tricolons(after)[:1]:
        problems.append(f"Still has a three-part list (“{triple}”).")
    sentences = split_sentences(after)
    cv = sentence_cv(sentences)
    if voice == "personal" and len(sentences) >= 2 and len(voice_markers(after)) < 2:
        problems.append(
            "It still sounds voiceless. Write it as the author: first person, a contraction, "
            "a short fragment or a direct question, and an opinion on what's solid and what's overstated."
        )
    if len(sentences) >= 3 and cv is not None and cv < TARGET_CV:
        lengths = ", ".join(str(len(s.split())) for s in sentences)
        problems.append(
            f"Sentence lengths are still too even ({lengths} words). Use one very short "
            "sentence (2–6 words) and one long one (25+)."
        )
    return lost, problems


# Generic claims → the question only the writer can answer (Phase 1).
_QUESTIONS = (
    (re.compile(r"\b(stud(?:y|ies)|research(?:ers)?|surveys?|reports?|data|evidence)\b", re.I),
     "Which study, report or source do you mean?"),
    (re.compile(r"\b(experts?|scientists?|analysts?|specialists?|professionals?)\b", re.I),
     "Who said it? A name, or the kind of person?"),
    (re.compile(r"\b(many|most|several|numerous|countless|a (?:growing )?number of|significant(?:ly)?|"
                r"increasing(?:ly)?|widespread)\b", re.I),
     "Roughly how many or how much? A figure you know is enough."),
    (re.compile(r"\b(recent(?:ly)?|nowadays|today|in recent years|modern)\b", re.I),
     "When, roughly? A year or period?"),
    (re.compile(r"\b(businesses|companies|organi[sz]ations|schools|cities|countries|communities|people|users|students)\b", re.I),
     "Which one? A place, company or group you have in mind?"),
)
_EXAMPLE_QUESTION = "One example you've seen or done yourself?"


def grounding_questions(passage: str, limit: int = 3) -> list[str]:
    """Up to ``limit`` short questions asking for the particulars a passage lacks.

    Zero tokens. Passages that already name things or give numbers get fewer
    questions; every answer is optional.
    """
    names, numbers = names_and_numbers(passage)
    questions = [q for pattern, q in _QUESTIONS if pattern.search(passage)]
    if numbers:
        questions = [q for q in questions if not q.startswith("Roughly")]
    if not (names or numbers) or len(questions) < limit:
        questions.append(_EXAMPLE_QUESTION)
    return list(dict.fromkeys(questions))[:limit]


def grounding_tips(text: str, limit: int = 3) -> list[dict]:
    """Paragraphs that make general claims with nothing specific (T3, §2.5).

    No model can honestly fix these: the remedy is the writer's own detail.
    """
    tips = []
    for index, paragraph in enumerate(p for p in _PARA_SPLIT_RE.split(text) if p.strip()):
        sentences = split_sentences(paragraph)
        if len(sentences) < 2:
            continue
        names, numbers = names_and_numbers(paragraph)
        if names or numbers:
            continue
        if not (_GENERIC_RE.search(paragraph) or lexicon_hits(paragraph)):
            continue
        opening = " ".join(WS_RE.sub(" ", paragraph).split()[:8])
        tips.append({"paragraph": index + 1, "opening": opening})
        if len(tips) >= limit:
            break
    return tips
