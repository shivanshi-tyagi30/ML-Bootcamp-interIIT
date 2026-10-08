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
    # "Prachi, should we..." (comma: any follow-up) / "Prachi can you..." (no comma: direct questions only).
    # "Standard whisper is terrible" is a statement, not someone being addressed.
    re.compile(r"(?:^|[.!?]\s+)" + NAME + r",\s+(?:can|could|will|would|should|shall|do|did|does|have|has|are|is|"
               r"what|how|why|when|where|which|please|you|any|tell|let's|go ahead|take over|take it)\b"),
    re.compile(r"(?:^|[.!?]\s+)" + NAME + r"\s+(?:can you|could you|will you|would you|please|do you|did you|"
               r"are you|have you|go ahead|take over)\b"),
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
    # Modal and helper verbs, pronouns and fillers that follow "Thanks," / "Hi," at the start of a question
    # ("Thanks, can you share the screen?") and must never be read as a name.
    "can", "could", "would", "should", "shall", "will", "might", "must", "do", "does", "did", "have", "has",
    "had", "please", "kindly", "maybe", "again", "everyone's", "me", "my", "your", "our", "us", "he", "she",
    "they", "them", "his", "her", "their", "its", "these", "those", "any", "some", "each", "every", "anyway",
    "though", "once", "first", "last", "finally", "quick", "quickly", "much", "lot", "guys", "ji", "bhai",
    "didi", "yaar", "dear", "buddy", "man", "boss", "mam", "ma'am", "welcome", "bye", "goodbye", "morning",
    "evening", "afternoon", "night", "noon", "thank", "listen", "look", "see", "wait", "excuse", "hold", "come",
    "tell", "let's", "lets", "hmm", "um", "uh", "oh", "ah", "alright",
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


def _find(patterns: list[re.Pattern[str]], text: str, allow_lower: bool | None = None) -> list[str]:
    """Names matched by any pattern (case-insensitive lead words, capitalized names).

    Lower-case candidates are accepted only for self-introductions ("my name is shivanshi"); after a
    greeting a lower-case word is almost always an ordinary word ("thanks, can you...").
    """
    if allow_lower is None:
        allow_lower = patterns is SELF_INTRO
    out = []
    for p in patterns:
        for m in re.finditer(p.pattern, text, flags=re.I):  # lead words in any case ("Hello", "hello")
            # Whisper normally capitalizes names. A lower-case candidate is kept (first word only) unless it
            # looks like an ordinary word; a lower-case second word is never part of the name.
            parts = m.group(1).split()
            if parts[0][0].isupper():
                words = [parts[0]] + ([parts[1]] if len(parts) > 1 and parts[1][0].isupper() else [])
            elif not allow_lower or NOT_NAME_SUFFIX.search(parts[0].lower()) or len(parts[0]) < 3:
                words = []
            else:
                words = [parts[0]]
            name = _clean(" ".join(words)) if words else None
            if name:
                out.append(name)
    return out


def find_speaker_names(segments: list[Segment]) -> dict[str, dict[str, Any]]:
    """{"Speaker 1": {"name": "Shivanshi", "evidence": [{"segment_id", "kind", "text"}]}} for named speakers."""
    # Lower-case names after greetings are trusted only when the transcript itself is lower-case (no
    # capitals at line starts); in a normally capitalized transcript a lower-case word there is not a name.
    starts = [t[0] for t in (s.text.strip() for s in segments) if t and t[0].isalpha()]
    lower_transcript = bool(starts) and sum(c.islower() for c in starts) / len(starts) > 0.5
    votes: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    proof: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for i, s in enumerate(segments):
        if not s.speaker:
            continue
        own = set()
        for name in dict.fromkeys(_find(SELF_INTRO, s.text)):  # one vote per line, however many rules match
            own.add(name)
            votes[s.speaker][name] += SELF_WEIGHT
            proof[(s.speaker, name)].append({"segment_id": s.id, "kind": "self_introduction", "text": s.text})
        for name in dict.fromkeys(_find(ADDRESS, s.text, allow_lower=lower_transcript)):
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
        a1 = _find(ADDRESS, s1.text, allow_lower=lower_transcript)
        a2 = _find(ADDRESS, s2.text, allow_lower=lower_transcript)
        if a1 and a2 and a1[0].lower() != a2[0].lower():
            name_a, name_b = a1[0], a2[0]
            # Speaker labels come from the voice model only; a greeting never rewrites them.
            if not (s1.speaker and s2.speaker and s1.speaker != s2.speaker):
                continue
            spk1, spk2 = s1.speaker, s2.speaker
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


SELF_CUE = re.compile(r"\b(?:i am|i'm|im|my name is|my name's|myself|this is|call me|it's)\s+$", re.I)
AFTER_SELF = re.compile(r"^\s*(?:here|speaking)\b", re.I)
LLM_REPLY_WINDOW = 3
# A name used to address someone: after a greeting or at the start of a sentence, and followed by a pause,
# the end, or a direct question ("Hi Sara, ...", "Hello Alex.", "Prachi can you...").
ADDRESSED_BEFORE = re.compile(r"(?:^|[.!?]\s*|\b(?:hi|hello|hey|thanks|thank you|bye|okay|ok|so|well|yes|no|"
                              r"morning|good morning|welcome|dear)[,!]?\s+|,\s*)$", re.I)
ADDRESSED_AFTER = re.compile(r"^(?:\s*[,.!?]|\s*$|\s+(?:can|could|will|would|do|did|are|have|please|what|how|why|"
                             r"when|where)\b)", re.I)


def verify_llm_names(proposed: list[Any], segments: list[Segment], taken: set[str] | None = None,
                     ) -> dict[str, dict[str, Any]]:
    """Keep only LM2's speaker names that the transcript supports. {label: {"name", "evidence"}}.

    The name must be said on the cited line and not be a common word, place or technical term, and the line
    must show who it is: the label introducing themselves ("I'm Sara", "Sara here"), someone addressing them
    just before they speak ("Hi Sara" -> Sara answers within the next few lines), or them being greeted right
    after they spoke ("Hello Alex" said to the person who spoke just before). One name per person.
    """
    by_id = {s.id: i for i, s in enumerate(segments)}
    labels = {s.speaker for s in segments if s.speaker}
    used = {n.lower() for n in (taken or set())}
    out: dict[str, dict[str, Any]] = {}
    for p in proposed:
        label = getattr(p, "label", "") or ""
        name = _clean(" ".join((getattr(p, "name", "") or "").split()[:2])) if getattr(p, "name", "") else None
        idx = by_id.get(getattr(p, "evidence_segment_id", "") or "")
        if (not name or label not in labels or not re.fullmatch(r"Speaker \d+", label) or label in out
                or name.lower() in used or idx is None):
            continue
        ev = segments[idx]
        m = re.search(r"(?<!\w)" + re.escape(name.split()[0]) + r"(?!\w)", ev.text, re.I)
        if not m:
            continue
        if ev.speaker == label:
            ok = bool(SELF_CUE.search(ev.text[: m.start()]) or AFTER_SELF.search(ev.text[m.end():]))
        elif not (ADDRESSED_BEFORE.search(ev.text[: m.start()]) and ADDRESSED_AFTER.search(ev.text[m.end():])):
            ok = False  # the word is there but not used to address someone ("Standard whisper is...")
        else:
            after = [s.speaker for s in segments[idx + 1: idx + 1 + LLM_REPLY_WINDOW]]
            before = next((s.speaker for s in reversed(segments[:idx]) if s.speaker and s.speaker != ev.speaker), None)
            ok = label in after or before == label
        if ok:
            out[label] = {"name": name, "evidence": [{"segment_id": ev.id, "kind": "model_verified", "text": ev.text}]}
            used.add(name.lower())
    return out


def rename_in_text(text: str, names: dict[str, dict[str, Any]]) -> str:
    """Replace "Speaker N" labels in model-written text with the verified names."""
    for label, v in names.items():
        text = re.sub(r"\b" + re.escape(label) + r"\b", v["name"], text)
    return text
