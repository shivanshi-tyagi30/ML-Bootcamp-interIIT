"""RAW_SAVED: build the raw transcript from words, re-check and speakers (spec Section 10.7)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from app.core.stages import Stage
from app.models.record import Segment, Word
from app.pipeline.recheck import flatten_words

if TYPE_CHECKING:
    from app.pipeline.runner import JobContext

PAUSE_SEC = 1.0
MAX_SEC, MAX_WORDS = 30.0, 60
SOFT_WORDS = 12  # inside a long Whisper segment, end on sentence punctuation after this many words
SENTENCE_END = re.compile(r"[.!?…]['\"”)]*$")


def speaker_for(word: dict[str, Any], turns: list[dict[str, Any]]) -> str | None:
    """Speaker whose turn overlaps the word most; else the nearest turn within 0.5 s."""
    if not turns:
        return None
    best, best_overlap = None, 0.0
    for t in turns:
        ov = min(word["end"], t["end"]) - max(word["start"], t["start"])
        if ov > best_overlap:
            best, best_overlap = t["speaker"], ov
    if best:
        return best
    mid = (word["start"] + word["end"]) / 2
    dist, near = min((min(abs(mid - t["start"]), abs(mid - t["end"])), t["speaker"]) for t in turns)
    return near if dist <= 0.5 else None


PSEUDO_PAUSE_SEC = 0.42  # natural turn pause in conversation (~0.4s)
MAX_PSEUDO_SPEAKERS = 3   # default typical meeting group size to rotate returning speakers


def detect_self_intro(text: str) -> str | None:
    """Detect name in self-introduction like 'I am Shivanshi', 'This is Prachi'."""
    m = re.search(r"\b(?:i am|i'm|my name is|this is)\s+([A-Z][a-z]+)\b", text, re.I)
    return m.group(1).capitalize() if m else None


def assign_pseudo_speakers(words: list[dict[str, Any]]) -> None:
    """Assign pseudo-speaker labels based on pauses and conversational turn-taking.
    
    Identifies turn boundaries (>0.4s pause), assigns distinct speakers, and re-identifies
    returning speakers rather than treating every turn as a new speaker.
    """
    if not words:
        return

    # First pass: identify word clusters / speech bursts separated by natural pauses
    bursts: list[list[dict[str, Any]]] = []
    cur_burst: list[dict[str, Any]] = [words[0]]
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap >= PSEUDO_PAUSE_SEC:
            bursts.append(cur_burst)
            cur_burst = []
        cur_burst.append(words[i])
    if cur_burst:
        bursts.append(cur_burst)

    # Detect if any bursts contain explicit self-introductions
    # e.g. "I am Shivanshi" -> assign named speaker or distinct speaker ID
    named_speakers: dict[str, int] = {}
    next_speaker_id = 1
    burst_speakers: list[int] = []

    # Count how many distinct introductory speakers appear
    for b in bursts:
        burst_text = " ".join(w["w"] for w in b)
        intro_name = detect_self_intro(burst_text)
        if intro_name:
            if intro_name not in named_speakers:
                named_speakers[intro_name] = next_speaker_id
                next_speaker_id += 1

    num_participants = max(2, min(MAX_PSEUDO_SPEAKERS, max(len(named_speakers), 2)))
    
    current_speaker = 1
    last_speaker = 1
    for idx, b in enumerate(bursts):
        burst_text = " ".join(w["w"] for w in b)
        intro_name = detect_self_intro(burst_text)
        
        if intro_name and intro_name in named_speakers:
            speaker_id = named_speakers[intro_name]
        elif idx == 0:
            speaker_id = 1
        else:
            # Alternating turn-taking: when the speaker changes, switch to the next active speaker
            # rather than creating an infinite sequence of new speakers.
            # Cycles through known participants so Speaker 1 speaks again!
            speaker_id = (last_speaker % num_participants) + 1

        last_speaker = speaker_id
        for w in b:
            w["speaker"] = f"Speaker {speaker_id}"


def join_words(words: list[str]) -> str:
    """Join words with single spaces (Whisper words already carry their punctuation)."""
    return re.sub(r"\s+", " ", " ".join(w.strip() for w in words)).strip()


def assemble_segments(
    whisper_segments: list[dict[str, Any]], word_updates: dict[str, dict[str, Any]], turns: list[dict[str, Any]],
) -> list[Segment]:
    """Raw transcript segments with ids S001.. in time order."""
    words = flatten_words(whisper_segments)
    for i, w in enumerate(words):
        w.update(word_updates.get(str(i), {}))
        
    if turns:
        for w in words:
            w["speaker"] = speaker_for(w, turns)
    else:
        assign_pseudo_speakers(words)

    groups: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    for w in words:
        if cur:
            prev = cur[-1]
            split = (
                w["speaker"] != prev["speaker"]
                or w["start"] - prev["end"] > PAUSE_SEC
                or w["end"] - cur[0]["start"] > MAX_SEC
                or len(cur) >= MAX_WORDS
                or (SENTENCE_END.search(prev["w"]) and (w["seg"] != prev["seg"] or len(cur) >= SOFT_WORDS))
            )
            if split:
                groups.append(cur)
                cur = []
        cur.append(w)
    if cur:
        groups.append(cur)

    segments = []
    for n, g in enumerate(groups, start=1):
        segments.append(Segment(
            id=f"S{n:03d}", start=round(g[0]["start"], 3), end=round(g[-1]["end"], 3), speaker=g[0]["speaker"],
            text=join_words([w["w"] for w in g]),
            words=[Word(w=w["w"], start=w["start"], end=w["end"], conf=w["conf"],
                        disputed=bool(w.get("disputed")), alt=w.get("alt")) for w in g],
        ))
    return segments


async def assemble_raw(ctx: "JobContext") -> None:
    """Stage function: write raw_transcript.json (never modified afterwards)."""
    whisper = ctx.read(ctx.output_name(Stage.TRANSCRIBING))
    recheck = ctx.read(ctx.output_name(Stage.RECHECKING)) or {}
    diar = ctx.read(ctx.output_name(Stage.DIARIZING)) or {}
    segs = assemble_segments(whisper["segments"], recheck.get("word_updates", {}), diar.get("turns", []))
    ctx.write(ctx.output_name(Stage.RAW_SAVED), [s.model_dump(mode="json") for s in segs])
