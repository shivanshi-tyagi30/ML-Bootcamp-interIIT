"""Shared text helpers: number/negation/modal patterns and normalization (spec Section 14)."""

from __future__ import annotations

import re
from collections import Counter

NUM_WORDS = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    "sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    "hundred|thousand|million|billion|half|quarter"
)
NUM = re.compile(rf"\d+(?:[.,:]\d+)*%?|\b(?:{NUM_WORDS})\b", re.I)
NEG = re.compile(r"\b(?:not|no|never|none|nor|nothing|nobody|cannot|without)\b|n't\b", re.I)
MODAL = re.compile(r"\b(?:will|won't|would|might|may|must|should|shall|can|can't|could|maybe|probably)\b", re.I)

# Number words with a single digit form, so "fifteen" and "15" compare equal.
_WORD_TO_DIGIT = {
    w: str(i)
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
        "fifteen sixteen seventeen eighteen nineteen".split()
    )
}
_WORD_TO_DIGIT.update(
    {"twenty": "20", "thirty": "30", "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
     "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000", "million": "1000000",
     "billion": "1000000000"}
)

FILLERS = {"um", "uh", "erm", "hmm", "mm", "ah", "like", "you know"}

SENTENCE_END = re.compile(r"[.!?:;\"'“”(\[]\s*$")


def normalize_token(t: str) -> str:
    """Lowercase, strip punctuation except apostrophes (curly ones become straight)."""
    t = t.replace("’", "'").replace("‘", "'").lower()
    return re.sub(r"[^\w']+", "", t).strip("'")


def normalize_number(t: str) -> str:
    """Canonical form of a number token: number words become digits, separators dropped."""
    t = normalize_token(t)
    if t in _WORD_TO_DIGIT:
        return _WORD_TO_DIGIT[t]
    return t.replace(",", "")


def numbers(text: str) -> Counter[str]:
    """Multiset of numbers in `text`, in canonical form."""
    return Counter(normalize_number(m.group(0)) for m in NUM.finditer(text))


def negation_count(text: str) -> int:
    """Number of negation tokens in `text`."""
    return len(NEG.findall(text))


def modal_set(text: str) -> set[str]:
    """Set of modal tokens in `text` (lowercase)."""
    return {m.lower().replace("’", "'") for m in MODAL.findall(text)}


def capitalized_non_initial(text: str) -> list[str]:
    """Capitalized tokens that are not sentence-initial: likely names or acronyms.

    Skips the pronoun "I" (and I'm/I'll...), and any token right after sentence
    punctuation, a colon or an opening quote.
    """
    out: list[str] = []
    for m in re.finditer(r"[A-Za-z][\w'’.-]*", text):
        tok = m.group(0).rstrip(".")
        if not tok[0].isupper():
            continue
        if re.fullmatch(r"I(?:['’](?:m|ll|d|ve))?", tok):
            continue
        before = text[: m.start()]
        if not before.strip() or SENTENCE_END.search(before):
            continue
        out.append(tok)
    return out


def estimate_tokens(text: str) -> int:
    """Rough token count (about 4 characters per token for English)."""
    return max(1, len(text) // 4)
