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


RESPONSE_CUES = re.compile(
    r"^(?:agreed|yes|yeah|yep|no|nope|sure|okay|ok|fine|right|exactly|that makes sense|let's postpone|let's finalize)\b",
    re.I,
)


def detect_self_intro(text: str) -> str | None:
    """Detect name in self-introduction like 'I am Shivanshi', 'This is Prachi'."""
    m = re.search(r"\b(?:i am|i'm|my name is|this is)\s+([A-Z][a-z]+)\b", text, re.I)
    return m.group(1).capitalize() if m else None


def assign_pseudo_speakers(words: list[dict[str, Any]]) -> None:
    """Assign pseudo-speaker labels based on conversational turn-taking and response cues.

    Identifies true speaker transitions (answers to questions, response signals like 'Agreed',
    and extended pauses) while keeping a speaker's pauses within their own turn from causing
    fake speaker switches.
    """
    if not words:
        return

    full_text = " ".join(w["w"] for w in words)
    named_mentions = set(re.findall(r"\b(?:hi|hello|hey)\s+([A-Z][a-z]+)\b", full_text, re.I))
    intros = set(re.findall(r"\b(?:i am|i'm|my name is|this is)\s+([A-Z][a-z]+)\b", full_text, re.I))
    all_names = {n.capitalize() for n in (named_mentions | intros)}
    num_participants = max(2, min(3, len(all_names) if len(all_names) >= 2 else 2))

    current_speaker = 1
    words[0]["speaker"] = f"Speaker {current_speaker}"

    for i in range(1, len(words)):
        prev = words[i - 1]
        cur = words[i]
        gap = cur["start"] - prev["end"]

        prev_ended_sentence = bool(re.search(r"[.!?…]['\"”)]*$", prev["w"]))
        prev_ended_question = bool(re.search(r"\?['\"”)]*$", prev["w"]))

        next_chunk = " ".join(words[k]["w"] for k in range(i, min(len(words), i + 4))).strip()
        is_response_cue = bool(RESPONSE_CUES.match(next_chunk))

        turn_changed = False
        if prev_ended_question and gap >= 0.35:
            turn_changed = True
        elif prev_ended_sentence and is_response_cue and gap >= 0.35:
            turn_changed = True
        elif prev_ended_sentence and gap >= 1.6:
            turn_changed = True
        elif gap >= 2.4:
            turn_changed = True

        if turn_changed:
            current_speaker = (current_speaker % num_participants) + 1

        cur["speaker"] = f"Speaker {current_speaker}"


def assign_speakers(words: list[dict[str, Any]], wav_path: Any = None) -> None:
    """Assign speaker labels using ECAPA-TDNN embedding clustering on audio bursts.
    
    Falls back gracefully to conversational pause heuristics if audio or model is unavailable.
    """
    if not words:
        return

    # Identify speech bursts separated by natural pauses
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

    # Try ECAPA-TDNN speaker embedding and clustering first
    if wav_path:
        try:
            from pathlib import Path
            if Path(wav_path).exists():
                from app.pipeline.ecapa_diarize import get_ecapa_identifier
                identifier = get_ecapa_identifier()
                spk_labels = identifier.identify_speakers(wav_path, bursts)
                if spk_labels and len(spk_labels) == len(bursts):
                    for b, spk in zip(bursts, spk_labels):
                        for w in b:
                            w["speaker"] = spk
                    return
        except Exception:  # noqa: BLE001
            pass

    # Fallback to pause-based pseudo speakers
    assign_pseudo_speakers(words)


def join_words(words: list[str]) -> str:
    """Join words with single spaces (Whisper words already carry their punctuation)."""
    return re.sub(r"\s+", " ", " ".join(w.strip() for w in words)).strip()


def assemble_segments(
    whisper_segments: list[dict[str, Any]],
    word_updates: dict[str, dict[str, Any]],
    turns: list[dict[str, Any]],
    wav_path: Any = None,
) -> list[Segment]:
    """Raw transcript segments with ids S001.. in time order.

    Groups a speaker's single dialogue turn into one line/segment.
    """
    words = flatten_words(whisper_segments)
    for i, w in enumerate(words):
        w.update(word_updates.get(str(i), {}))

    if turns:
        for w in words:
            w["speaker"] = speaker_for(w, turns)
    else:
        assign_speakers(words, wav_path=wav_path)

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
    segs = assemble_segments(
        whisper["segments"],
        recheck.get("word_updates", {}),
        diar.get("turns", []),
        wav_path=ctx.wav_path,
    )
    ctx.write(ctx.output_name(Stage.RAW_SAVED), [s.model_dump(mode="json") for s in segs])
