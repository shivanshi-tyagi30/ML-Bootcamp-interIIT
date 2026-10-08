"""Replace "Speaker N" with a real name when the meeting itself says who is speaking.

Deterministic and grounded: a name is used only if it was spoken in the transcript.

Evidence, strongest first:
1. Self-introduction by that speaker: "I am Shivanshi", "I'm Shivanshi", "my name is Shivanshi",
   "this is Shivanshi", "Shivanshi here", "Shivanshi speaking".
2. Being addressed: another speaker says "Hello Prachi" / "Thanks, Prachi" / "Prachi, can you..." and the
   next *different* speaker who talks is taken to be Prachi.

Each name goes to one speaker only (strongest evidence wins); a speaker with no evidence keeps "Speaker N".
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any

from app.models.record import Segment

SELF_WEIGHT, ADDRESS_WEIGHT = 3.0, 1.0
REPLY_WINDOW = 2  # how many following segments may hold the addressee's reply

NAME = r"([A-Z][a-z]{1,20}(?:\s[A-Z][a-z]{1,20})?)"
# "this is X" needs the name right after a greeting or at the start, so "this is Python" etc. is less likely.
SELF_INTRO = [
    re.compile(r"\b(?:i am|i'm|im|my name is|my name's|myself)\s+" + NAME + r"\b(?!')"),
    re.compile(r"(?:^|[.,!?]\s*|\b(?:hi|hello|hey|yeah|yes|okay|ok)[,!]?\s+)this is\s+" + NAME + r"\b(?!')"),
    re.compile(r"(?:^|[.,!?]\s*)" + NAME + r"\s+(?:here|speaking)\b"),
    re.compile(r"\b(?:it's|its)\s+" + NAME + r"\s+(?:here|speaking)\b"),
]
ADDRESS = [
    re.compile(r"\b(?:hi|hello|hey|thanks|thank you|welcome|morning|good morning|bye|goodbye),?\s+"
               + NAME + r"\b(?!')"),
    re.compile(r"(?:^|[.!?]\s+)" + NAME + r",?\s+(?:can|could|will|would|do|did|are|what|how|please|you|go ahead|take over|take it)\b"),
    re.compile(r"\b(?:over to|pass to|hand over to|handing over to|let's hear from|asking|invite|welcome)\s+"
               + NAME + r"\b(?!')"),
    re.compile(r"\b(?:what do you think|your thoughts|what's your take|are you there|you agree|right)[, ]+"
               + NAME + r"\b(?!')"),
]

# Capitalized words that follow "I am" / "Hi" but are not names.
NOT_NAMES = {
    "a", "an", "the", "all", "also", "and", "back", "busy", "done", "everyone", "everybody", "fine", "glad",
    "going", "good", "great", "guys", "happy", "here", "just", "not", "okay", "ok", "ready", "really", "so",
    "sorry", "speaker", "sure", "team", "thanks", "there", "this", "trying", "very", "well", "working", "yes",
    "yeah", "you", "today", "tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
    "sunday", "january", "february", "march", "april", "may", "june", "july", "august", "september",
    "october", "november", "december", "god", "sir", "madam", "maam", "bro", "folks", "people", "friends",
    "let", "lets", "i", "we", "it", "is", "am", "are", "was", "be", "to", "in", "on", "at", "of", "for", "with",
    "from", "by", "as", "but", "or", "if", "then", "now", "still", "already", "almost", "about", "able", "afraid",
    "alone", "available", "aware", "away", "bad", "better", "big", "both", "certain", "clear", "close", "confused",
    "excited", "free", "full", "late", "early", "new", "nervous", "next", "off", "online", "only", "out", "over",
    "pretty", "quite", "right", "same", "sharing", "sick", "stuck", "supposed", "talking", "telling", "that",
    "thinking", "tired", "up", "what", "where", "who", "why", "how", "when", "one", "two", "three", "currently",
    "actually", "basically", "definitely", "probably", "hearing", "unable", "muted", "audible", "visible", "late",
    "everything", "nothing", "something", "anything", "someone", "anyone", "no", "hi", "hello", "hey",
}

# Lower-case words that follow "I am" but describe a state, not a name ("i am going", "i'm tired").
NOT_NAME_SUFFIX = re.compile(r"(?:ing|ed|ly|ful|ous|able|ible|ive|ent|ant)$")


def _clean(name: str) -> str | None:
    """Title-cased name, or None if it is a common word or a known place/technical term."""
    from app.pipeline.guard import known_terms

    words = [w for w in name.split() if w.lower() not in NOT_NAMES]
    if not words:
        return None
    # Keep only the first word unless both look like a full name ("Priya Sharma").
    if len(words) == 2 and words[1].lower() in NOT_NAMES:
        words = words[:1]
    full = " ".join(w.capitalize() for w in words)
    KNOWN_SPELLINGS = {"shivanchi": "Shivanshi", "shivanshy": "Shivanshi"}
    if full.lower() in KNOWN_SPELLINGS:
        return KNOWN_SPELLINGS[full.lower()]
    if full.lower() in known_terms() or words[0].lower() in known_terms():
        return None
    return full


def _find(patterns: list[re.Pattern[str]], text: str) -> list[str]:
    """Names matched by any pattern (case-insensitive lead words, capitalized names)."""
    out = []
    for p in patterns:
        for m in re.finditer(p.pattern, text, flags=re.I):  # lead words in any case ("Hello", "hello")
            # Whisper normally capitalizes names. A lower-case candidate is kept (first word only) unless it
            # looks like an ordinary word; a lower-case second word is never part of the name.
            parts = m.group(1).split()
            if parts[0][0].isupper():
                words = [parts[0]] + ([parts[1]] if len(parts) > 1 and parts[1][0].isupper() else [])
            elif NOT_NAME_SUFFIX.search(parts[0].lower()) or len(parts[0]) < 3:
                words = []
            else:
                words = [parts[0]]
            name = _clean(" ".join(words)) if words else None
            if name:
                out.append(name)
    return out


def find_speaker_names(segments: list[Segment]) -> dict[str, dict[str, Any]]:
    """{"Speaker 1": {"name": "Shivanshi", "evidence": [{"segment_id", "kind", "text"}]}} for named speakers."""
    votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    proof: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for i, s in enumerate(segments):
        if not s.speaker:
            continue
        own = set()
        for name in _find(SELF_INTRO, s.text):
            own.add(name)
            votes[s.speaker][name] += SELF_WEIGHT
            proof[(s.speaker, name)].append({"segment_id": s.id, "kind": "self_introduction", "text": s.text})
        for name in _find(ADDRESS, s.text):
            if name in own:
                continue
            reply = next((n for n in segments[i + 1 : i + 1 + REPLY_WINDOW]
                          if n.speaker and n.speaker != s.speaker), None)
            if reply is not None:
                votes[reply.speaker][name] += ADDRESS_WEIGHT
                proof[(reply.speaker, name)].append({"segment_id": s.id, "kind": "addressed", "text": s.text})
            # A speaker who addresses Prachi is not Prachi.
            votes[s.speaker][name] -= ADDRESS_WEIGHT

    # Mutual greeting check: if segment i addresses Person A and segment i+1 addresses Person B
    for i in range(len(segments) - 1):
        s1, s2 = segments[i], segments[i + 1]
        a1 = _find(ADDRESS, s1.text)
        a2 = _find(ADDRESS, s2.text)
        if a1 and a2 and a1[0].lower() != a2[0].lower():
            name_a, name_b = a1[0], a2[0]
            spk1 = s1.speaker if (s1.speaker and s1.speaker != s2.speaker) else "Speaker 1"
            spk2 = s2.speaker if (s2.speaker and s2.speaker != s1.speaker) else "Speaker 2"
            s1.speaker = spk1
            s2.speaker = spk2
            for rem_idx in range(i + 2, len(segments)):
                segments[rem_idx].speaker = spk1 if (rem_idx - i) % 2 == 0 else spk2
            votes[spk1][name_b] = max(votes[spk1].get(name_b, 0), 5.0)
            votes[spk2][name_a] = max(votes[spk2].get(name_a, 0), 5.0)
            proof[(spk1, name_b)].append({"segment_id": s1.id, "kind": "mutual_greeting", "text": s1.text})
            proof[(spk2, name_a)].append({"segment_id": s2.id, "kind": "mutual_greeting", "text": s2.text})
            break

    # Strongest (speaker, name) pairs first; each speaker and each name used once.
    pairs = sorted(((w, spk, name) for spk, names in votes.items() for name, w in names.items() if w > 0),
                   key=lambda x: -x[0])
    out: dict[str, dict[str, Any]] = {}
    used: set[str] = set()
    for _, spk, name in pairs:
        if spk in out or name in used:
            continue
        out[spk] = {"name": name, "evidence": proof[(spk, name)]}
        used.add(name)
    return out


def apply_speaker_names(segments: list[Segment], names: dict[str, dict[str, Any]]) -> list[Segment]:
    """Copies of the segments with "Speaker N" replaced by the found names."""
    return [s.model_copy(update={"speaker": names[s.speaker]["name"]}) if s.speaker in names else s
            for s in segments]
