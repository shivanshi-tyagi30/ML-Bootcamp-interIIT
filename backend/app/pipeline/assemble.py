"""RAW_SAVED: build the raw transcript from words, re-check and speakers (spec Section 10.7)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from app.core.stages import Stage
from app.models.record import Segment, Word
from app.pipeline.recheck import flatten_words
from app.pipeline.speaker_names import apply_speaker_names, find_speaker_names

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


def join_words(words: list[str]) -> str:
    """Join words with single spaces (Whisper words already carry their punctuation)."""
    return re.sub(r"\s+", " ", " ".join(w.strip() for w in words)).strip()


def assemble_segments(
    whisper_segments: list[dict[str, Any]], word_updates: dict[str, dict[str, Any]], turns: list[dict[str, Any]],
) -> list[Segment]:
    """Raw transcript segments with ids S001.. in time order.

    Speakers come only from the DIARIZING stage; without turns they stay empty (never guessed).
    """
    words = flatten_words(whisper_segments)
    for i, w in enumerate(words):
        w.update(word_updates.get(str(i), {}))
        w["speaker"] = speaker_for(w, turns)

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
                or (SENTENCE_END.search(prev["w"]) and (w.get("seg") != prev.get("seg") or len(cur) >= SOFT_WORDS))
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
    if ctx.settings.SPEAKER_NAMES_FROM_TRANSCRIPT:
        # "Hello Prachi, I am Shivanshi" -> Speaker 1 becomes Shivanshi everywhere (evidence kept on disk).
        names = find_speaker_names(segs)
        if names:
            segs = apply_speaker_names(segs, names)
            ctx.log.info("speaker names: %s", {k: v["name"] for k, v in names.items()})
        ctx.write("speaker_names.json", names)
    ctx.write(ctx.output_name(Stage.RAW_SAVED), [s.model_dump(mode="json") for s in segs])
