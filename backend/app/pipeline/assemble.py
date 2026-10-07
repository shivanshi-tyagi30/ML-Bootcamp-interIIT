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


PSEUDO_PAUSE_SEC = 1.5  # gap between words that suggests a speaker change
MAX_PSEUDO_SPEAKERS = 6  # cap to avoid "Speaker 47" in long recordings


def assign_pseudo_speakers(words: list[dict[str, Any]]) -> None:
    """Assign pseudo-speaker labels based on pauses when diarization is unavailable.
    
    Uses significant pauses between words to guess speaker changes.
    Labels cycle through Speaker 1..N. This is a rough heuristic.
    """
    if not words:
        return
    speaker_num = 1
    words[0]["speaker"] = f"Speaker {speaker_num}"
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap > PSEUDO_PAUSE_SEC:
            speaker_num = (speaker_num % MAX_PSEUDO_SPEAKERS) + 1
        words[i]["speaker"] = f"Speaker {speaker_num}"


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
