"""VERIFYING: deterministic checks that can only remove or downgrade (spec Section 10.12)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from rapidfuzz import fuzz

from app.config import Settings
from app.core.stages import Stage
from app.core.text import capitalized_non_initial, numbers
from app.models.llm_io import LM2Output
from app.models.record import (
    UNSPECIFIED, ActionItem, CitedSentence, Decision, Fidelity, MeetingRecord, Meta, MinutesTopic, Pointer,
    Proposal, Refinement, Segment,
)
from app.pipeline.fidelity import compute_fidelity
from app.pipeline.guard import _word_spans

if TYPE_CHECKING:
    from app.pipeline.runner import JobContext

AGREEMENT_CUE = re.compile(
    r"\b(?:agree|agreed|decided|decision|final|let's go|go with|we will|we'll|approved|confirmed|settled|"
    r"done deal|sounds good|yes,? let's)\b", re.I,
)
INTRO = r"\b(?:this is|i am|i'm|my name is|it's)\s+"
DEADLINE_ADJACENCY = 2


@dataclass
class VerifyStats:
    """What the verifier changed."""

    sentences_removed: int = 0
    items_dropped: int = 0
    decisions_demoted: int = 0


def words_disputed(seg: Segment, exact: str) -> bool:
    """Whether any word covering `exact` in the segment is marked disputed."""
    i = seg.text.lower().find(exact.lower())
    if i < 0:
        return False
    j = i + len(exact)
    return any(a < j and i < b and w.disputed for (a, b), w in zip(_word_spans(seg), seg.words))


def resolve_pointer(
    ev: Pointer | None, item_seg_ids: list[str], seg: dict[str, Segment], diarization_on: bool,
    kind: str = "owner", order: dict[str, int] | None = None,
) -> tuple[str, str | None]:
    """Text copied from the transcript at the pointer, or (Unspecified, reason flag)."""
    if ev is None:
        return UNSPECIFIED, None
    s = seg.get(ev.segment_id)
    if s is None or not ev.exact_words.strip() or ev.exact_words.lower() not in s.text.lower():
        return UNSPECIFIED, "pointer_invalid"
    if words_disputed(s, ev.exact_words):
        return UNSPECIFIED, "audio_unclear"
    if ev.segment_id not in item_seg_ids:
        if kind == "deadline":
            order = order or {}
            near = any(
                abs(order.get(ev.segment_id, -99) - order.get(i, 99)) <= DEADLINE_ADJACENCY for i in item_seg_ids
            )
            if not near:
                return UNSPECIFIED, "pointer_invalid"
        else:  # cross-segment owner = self-introduction case
            if not diarization_on:
                return UNSPECIFIED, "self_assignment_unverified"
            intro = re.search(INTRO + re.escape(ev.exact_words), s.text, re.I)
            same_speaker = any(seg[i].speaker and seg[i].speaker == s.speaker for i in item_seg_ids if i in seg)
            if not (intro and same_speaker):
                return UNSPECIFIED, "self_assignment_unverified"
    i = s.text.lower().index(ev.exact_words.lower())
    return s.text[i : i + len(ev.exact_words)], None


def unsupported_facts(sentence: str, cited_text: str) -> list[str]:
    """Numbers and likely names in `sentence` that do not appear in `cited_text`."""
    missing = [n for n in numbers(sentence) if n not in numbers(cited_text)]
    low = cited_text.lower()
    for tok in capitalized_non_initial(sentence):
        if not re.search(r"(?<!\w)" + re.escape(tok.lower()) + r"(?!\w)", low):
            missing.append(tok)
    return missing


def _quote_matches(quote: str, ids: list[str], seg: dict[str, Segment], threshold: int) -> bool:
    """Whether `quote` fuzzy-matches the text of a cited segment (or all of them joined)."""
    q = quote.lower().strip()
    if not q:
        return False
    texts = [seg[i].text.lower() for i in ids if i in seg]
    return any(fuzz.partial_ratio(q, t) >= threshold for t in [*texts, " ".join(texts)])


def verify(
    lm2: LM2Output, raw: list[Segment], refined: list[Segment], refinement: Refinement, meta: Meta,
    settings: Settings, diarization_on: bool,
) -> MeetingRecord:
    """Apply the spec's checks in order and assemble the final record."""
    seg = {s.id: s for s in refined}
    order = {s.id: i for i, s in enumerate(refined)}
    stats = VerifyStats()
    known = lambda ids: [i for i in dict.fromkeys(ids) if i in seg]  # noqa: E731
    cited_text = lambda ids: " ".join(seg[i].text for i in ids)  # noqa: E731
    thr = settings.QUOTE_MATCH_THRESHOLD

    # 1-2. Summary and minutes: known ids only, no new facts.
    def keep_sentence(s: CitedSentence) -> CitedSentence | None:
        ids = known(s.evidence_segment_ids)
        if not ids or unsupported_facts(s.text, cited_text(ids)):
            stats.sentences_removed += 1
            return None
        return CitedSentence(text=s.text, evidence_segment_ids=ids)

    summary = [x for x in map(keep_sentence, lm2.summary) if x]
    minutes = []
    for t in lm2.minutes:
        pts = [x for x in map(keep_sentence, t.points) if x]
        if pts:
            minutes.append(MinutesTopic(topic=t.topic, points=pts))

    # 3. Decisions need an agreement cue that matches a cited segment.
    decisions: list[Decision] = []
    proposals: list[Proposal] = []
    for p in lm2.open_proposals:
        ids = known(p.evidence_segment_ids)
        if ids:
            proposals.append(Proposal(proposal=p.proposal, evidence_segment_ids=ids))
        else:
            stats.items_dropped += 1
    for d in lm2.decisions:
        ids = known(d.evidence_segment_ids)
        if not ids:
            stats.items_dropped += 1
            continue
        if AGREEMENT_CUE.search(d.agreement_evidence) and _quote_matches(d.agreement_evidence, ids, seg, thr):
            decisions.append(Decision(id="D0", decision=d.decision, agreement_evidence=d.agreement_evidence,
                                      evidence_segment_ids=ids))
        else:
            stats.decisions_demoted += 1
            proposals.append(Proposal(proposal=d.decision, evidence_segment_ids=ids, demoted_from_decision=True))

    # 4-5. Action items: quote must match; owner/deadline only from the transcript.
    tasks: list[ActionItem] = []
    for t in lm2.action_items:
        ids = known(t.evidence_segment_ids)
        if not ids or not _quote_matches(t.evidence_quote, ids, seg, thr):
            stats.items_dropped += 1
            continue
        flags: list[str] = []
        owner, why = resolve_pointer(t.owner_evidence, ids, seg, diarization_on, "owner", order)
        if owner == UNSPECIFIED and t.owner_evidence is not None:
            flags += ["owner_downgraded", *([why] if why else [])]
        deadline, why = resolve_pointer(t.deadline_evidence, ids, seg, diarization_on, "deadline", order)
        if deadline == UNSPECIFIED and t.deadline_evidence is not None:
            flags += ["deadline_downgraded", *([why] if why else [])]
        tasks.append(ActionItem(
            id="T0", task=t.task, owner=owner, owner_evidence=t.owner_evidence, deadline=deadline,
            deadline_evidence=t.deadline_evidence, evidence_quote=t.evidence_quote, evidence_segment_ids=ids,
            verifier_flags=list(dict.fromkeys(flags)),  # type: ignore[arg-type]
        ))

    # 6. Renumber in transcript order.
    first = lambda ids: min(order[i] for i in ids)  # noqa: E731
    decisions.sort(key=lambda d: first(d.evidence_segment_ids))
    tasks.sort(key=lambda t: first(t.evidence_segment_ids))
    proposals.sort(key=lambda p: first(p.evidence_segment_ids))
    for n, d in enumerate(decisions, 1):
        d.id = f"D{n}"
    for n, t in enumerate(tasks, 1):
        t.id = f"T{n}"

    # 7. Fidelity and the final record.
    rec = MeetingRecord(
        meta=meta, raw_transcript=raw, refined_transcript=refined, refinement=refinement, summary=summary,
        minutes=minutes, decisions=decisions, open_proposals=proposals, action_items=tasks,
        fidelity=Fidelity(numbers_preserved="0/0", negations_preserved="0/0", edits_accepted=0, edits_rejected=0,
                          disputed_words=0, items_downgraded_by_verifier=0, sentences_removed_by_verifier=0,
                          transcript_coverage_pct=0),
    )
    rec.fidelity = compute_fidelity(rec, refinement, stats.items_dropped, stats.decisions_demoted,
                                    stats.sentences_removed)
    return MeetingRecord.model_validate(rec.model_dump())


def build_meta(ctx: "JobContext") -> Meta:
    """Meta from the job's stage files and settings."""
    whisper = ctx.read(ctx.output_name(Stage.TRANSCRIBING)) or {}
    recheck = ctx.read(ctx.output_name(Stage.RECHECKING)) or {}
    diar = ctx.read(ctx.output_name(Stage.DIARIZING)) or {}
    probe = ctx.read(ctx.output_name(Stage.NORMALIZING)) or {}
    s = ctx.settings
    return Meta(
        job_id=ctx.job_id, title=ctx.title, source_file=ctx.upload.get("source_file", ""),
        duration_s=float(probe.get("duration_s") or 0), language=whisper.get("language", "en"),
        language_probability=float(whisper.get("language_probability", 1.0)),
        models={
            "stt": whisper.get("model", f"faster-whisper {s.WHISPER_MODEL}"),
            "stt_check": recheck.get("model") or (
                f"skipped: {recheck.get('reason', 'n/a')}"
                + ("; low-confidence words marked disputed instead" if recheck.get("fallback") else "")
            ),
            "diarization": diar.get("model") or f"skipped: {diar.get('reason', 'off')}",
            "lm1": s.LM1_MODEL,
            "lm2": s.LM2_MODEL,
        },
        warnings=list(whisper.get("warnings", [])),
        generated_at=datetime.now(timezone.utc),
    )


async def verify_record(ctx: "JobContext") -> None:
    """Stage function: write record.json and update the job row."""
    raw = [Segment(**s) for s in ctx.read(ctx.output_name(Stage.RAW_SAVED))]
    refined = [Segment(**s) for s in ctx.read("refined_transcript.json")]
    refinement = Refinement(**ctx.read(ctx.output_name(Stage.GUARDING)))
    lm2 = LM2Output(**ctx.read(ctx.output_name(Stage.DOCUMENTING)))
    diar = ctx.read(ctx.output_name(Stage.DIARIZING)) or {}
    rec = verify(lm2, raw, refined, refinement, build_meta(ctx), ctx.settings, diarization_on=not diar.get("skipped", True))
    ctx.write(ctx.output_name(Stage.VERIFYING), rec)
    await ctx.db.update(ctx.job_id, n_decisions=len(rec.decisions), n_tasks=len(rec.action_items),
                        duration_s=rec.meta.duration_s)
