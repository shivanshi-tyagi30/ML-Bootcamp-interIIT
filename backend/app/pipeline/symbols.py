"""Spoken symbols in technical names: "user underscore id" -> user_id, "at the rate frontend" -> @frontend.

Speech-to-text writes symbols as the words that were said. These deterministic rules turn them into the
written form in the REFINED transcript only (the raw transcript stays verbatim). Every change is recorded as
an applied edit (category "symbol"), so it is visible in the diff view like any other refinement.
"""

from __future__ import annotations

import re

from app.models.record import Edit, Segment

EXTENSIONS = (
    "com|in|org|net|io|ai|dev|app|co|edu|gov|uk|us|me|info|py|js|ts|tsx|jsx|json|md|txt|csv|html|css|yaml|yml|"
    "env|sh|ipynb|pdf|docx|xlsx|pptx|png|jpg|jpeg|svg|zip|exe|java|cpp|go|rs|sql|toml|cfg|ini|lock|log"
)
AT = r"(?:at the rate|at sign|at symbol)"

# (pattern, replacement, rationale). Applied repeatedly so chains work ("a underscore b underscore c").
RULES: list[tuple[re.Pattern[str], str, str]] = [
    (re.compile(r"\b([A-Za-z0-9_]+) underscore ([A-Za-z0-9_]+)\b", re.I), r"\1_\2", "spoken underscore"),
    (re.compile(rf"\b([A-Za-z0-9_-]+) dot ((?:[A-Za-z0-9-]+\.)*(?:{EXTENSIONS}))\b", re.I), r"\1.\2",
     "spoken dot in a file or web name"),
    # e-mail: "shivanshi at the rate gmail.com" (after the dot rule) -> shivanshi@gmail.com
    (re.compile(rf"\b([A-Za-z0-9_.]+) {AT} ([A-Za-z0-9-]+(?:\.[A-Za-z]{{2,}})+)\b", re.I), r"\1@\2",
     "spoken at sign in an e-mail address"),
    # mention / handle: "at the rate frontend" -> @frontend ("at the rate of 5 percent" is a real rate)
    (re.compile(rf"\b{AT} (?!of\b)([A-Za-z][A-Za-z0-9_.-]*)", re.I), r"@\1", "spoken at sign"),
    (re.compile(r"\bhash ?tag ([A-Za-z][A-Za-z0-9_]*)", re.I), r"#\1", "spoken hashtag"),
]
MAX_PASSES = 30


def symbol_edits(seg: Segment) -> tuple[Segment, list[Edit]]:
    """The segment with spoken symbols written out, and the edits that did it."""
    from app.pipeline.guard import apply_to_segment

    edits: list[Edit] = []
    for _ in range(MAX_PASSES):
        for pattern, repl, why in RULES:
            m = pattern.search(seg.text)
            if m:
                e = Edit(segment_id=seg.id, original=m.group(0), replacement=m.expand(repl), category="symbol",
                         rationale=why, confidence=1.0)
                # apply_to_segment locates an edit by its first occurrence; this match is the first one too.
                seg = apply_to_segment(seg, [e])
                edits.append(e)
                break
        else:
            break
    return seg, edits


def write_symbols(segments: list[Segment]) -> tuple[list[Segment], list[Edit]]:
    """Apply `symbol_edits` to every segment."""
    out, edits = [], []
    for s in segments:
        s2, e = symbol_edits(s)
        out.append(s2)
        edits += e
    return out, edits
