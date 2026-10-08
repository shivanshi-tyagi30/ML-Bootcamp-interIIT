"""Code finds the lines that look like tasks, decisions and proposals; LM2 then judges them.

Small models miss items in a long transcript when asked to "find everything". A cheap keyword scan gives
LM2 a checklist of the lines worth a close look, and the verifier uses the same cues to repair quotes
the model paraphrased. The scan only points; it never creates a record item on its own.
"""

from __future__ import annotations

import re
from typing import Any

from app.models.record import Segment

TASK_CUE = re.compile(
    r"\b(?:i'll|i will|i can|i'm going to|i am going to|let me|i'll take|i will take|i'll handle|"
    r"can you|could you|would you|will you|please|you need to|we need to|need to|needs to|has to|have to|"
    r"make sure|follow up|action item|assign(?:ed)?|take care of|take that|on it|i'll do|i will do|"
    r"responsible for|in charge of|by (?:today|tomorrow|tonight|monday|tuesday|wednesday|thursday|friday|"
    r"saturday|sunday|next|the end|end of|eod|eow|\d))\b",
    re.I,
)
DECISION_CUE = re.compile(
    r"\b(?:agreed|we agree|decided|decision|final call|finali[sz]e[d]?|let's go with|we'll go with|"
    r"we will go with|go ahead|approved|confirmed|settled|that's final|done deal|let's do (?:it|that)|"
    r"let's postpone|we'll postpone|we will postpone|postponed|we won't|we will not|we're not going to|"
    r"we are not going to|for now we'll|for now we will|we'll use|we will use|we'll keep|we will keep)\b",
    re.I,
)
PROPOSAL_CUE = re.compile(
    r"\b(?:maybe|perhaps|we could|we can also|what if|should we|shall we|how about|i suggest|i propose|"
    r"one option|alternatively|could we|why don't we|why not|we might|it might be|i think we should)\b",
    re.I,
)
# The speaker commits to doing it themselves.
FIRST_PERSON_COMMIT = re.compile(
    r"\b(?:i'll|i will|i can|let me|i'm going to|i am going to|leave it to me|i'm on it|i am on it|on it|"
    r"i'll take|i'll handle|i'll do)\b", re.I,
)
REQUEST = re.compile(r"\b(?:can you|could you|would you|will you|please)\b", re.I)

# A short reply from another person that accepts what was just proposed.
REPLY_AGREE = re.compile(
    r"^\W*(?:agreed|yes|yeah|yep|sure|sounds good|fine by me|makes sense|works for me|okay,? let's|"
    r"ok,? let's|let's do (?:it|that)|done|absolutely|definitely|perfect|great,? let's)\b",
    re.I,
)
REPLY_MAX_WORDS = 12


def is_agreeing_reply(prev: Segment, reply: Segment) -> bool:
    """`reply` is a short acceptance said by a different speaker than `prev`."""
    different = not (prev.speaker and reply.speaker and prev.speaker == reply.speaker)
    return different and len(reply.text.split()) <= REPLY_MAX_WORDS and bool(REPLY_AGREE.search(reply.text))


def find_candidates(segments: list[Segment]) -> list[dict[str, Any]]:
    """[{"ids": [...], "kinds": ["task" | "decision" | "proposal"], "text": "..."}] in transcript order."""
    out: list[dict[str, Any]] = []
    for i, s in enumerate(segments):
        kinds = []
        if TASK_CUE.search(s.text):
            kinds.append("task")
        if DECISION_CUE.search(s.text):
            kinds.append("decision")
        if PROPOSAL_CUE.search(s.text):
            kinds.append("proposal")
        ids = [s.id]
        nxt = segments[i + 1] if i + 1 < len(segments) else None
        if nxt is not None and is_agreeing_reply(s, nxt) and (kinds or "?" in s.text):
            # "Should we ship v2?" + "Agreed." from someone else: a likely decision (or an accepted task).
            ids.append(nxt.id)
            kinds = [k for k in kinds if k != "proposal"] + ["decision" if "task" not in kinds else "task"]
        if kinds:
            out.append({"ids": ids, "kinds": list(dict.fromkeys(kinds)), "text": s.text})
    return out


def checklist(candidates: list[dict[str, Any]], segment_ids: set[str] | None = None) -> str:
    """The CHECKLIST block for LM2 (optionally only for lines inside one chunk)."""
    rows = [c for c in candidates if segment_ids is None or c["ids"][0] in segment_ids]
    if not rows:
        return ""
    lines = [f"- [{', '.join(c['ids'])}] looks like: {' / '.join(c['kinds'])}" for c in rows]
    return (
        "\n\nCHECKLIST (a keyword search flagged these lines; it is often wrong). For EACH line decide whether "
        "it is an agreed decision, an open proposal, an action item, or nothing, using the DEFINITIONS. "
        "Also add real items the search missed.\n" + "\n".join(lines)
    )
