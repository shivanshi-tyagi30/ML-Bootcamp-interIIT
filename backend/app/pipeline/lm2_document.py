"""DOCUMENTING: LM2 writes the record with pointers and citations (spec Section 10.11)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from rapidfuzz import fuzz

from app.core.errors import PipelineError
from app.core.stages import Stage
from app.core.text import estimate_tokens
from app.llm.client import LLMUnavailable, load_prompt
from app.llm.json_repair import InvalidModelOutput
from app.models.llm_io import LM2Output, LM2Record, LM2Topic, SummaryPick
from app.pipeline.lm1_vocabulary import chunk_lines, transcript_lines

if TYPE_CHECKING:
    from app.pipeline.runner import JobContext

DEDUPE_RATIO = 90


NO_SCRATCHPAD_NOTE = (
    "\n\nNOTE: This output has no scratchpad field. Do the PROCEDURE silently in your head and output "
    "only the record fields."
)


def user_message(lines: list[str], schema_model: type = LM2Output) -> str:
    """Transcript plus the JSON schema (helps servers without constrained decoding)."""
    schema = json.dumps(schema_model.model_json_schema(), ensure_ascii=False)
    return "TRANSCRIPT\n" + "\n".join(lines) + "\n\nJSON SCHEMA\n" + schema


def overlapping_chunks(lines: list[str], max_tokens: int) -> list[list[str]]:
    """Chunks on segment boundaries, each starting with ~10% of the previous chunk."""
    base = chunk_lines(lines, int(max_tokens * 0.9))
    out = [base[0]] if base else []
    for prev, cur in zip(base, base[1:]):
        overlap = prev[-max(1, len(prev) // 10) :]
        out.append(overlap + cur)
    return out


def _dedupe(items: list[Any], text_of, ids_of) -> list[Any]:
    """Drop near-duplicate items (similar text and shared evidence), merging their evidence ids."""
    kept: list[Any] = []
    for it in items:
        twin = next(
            (k for k in kept
             if fuzz.token_set_ratio(text_of(k), text_of(it)) >= DEDUPE_RATIO and set(ids_of(k)) & set(ids_of(it))),
            None,
        )
        if twin is None:
            kept.append(it)
        else:
            ids_of(twin).extend(i for i in ids_of(it) if i not in ids_of(twin))
    return kept


def merge_outputs(parts: list[LM2Record]) -> LM2Output:
    """Merge per-chunk outputs; the summary is chosen afterwards."""
    topics: dict[str, LM2Topic] = {}
    for p in parts:
        for t in p.minutes:
            key = t.topic.strip().lower()
            if key in topics:
                have = {x.text for x in topics[key].points}
                topics[key].points.extend(x for x in t.points if x.text not in have)
            else:
                topics[key] = t.model_copy(deep=True)
    return LM2Output(
        summary=[s for p in parts for s in p.summary],
        minutes=list(topics.values()),
        decisions=_dedupe([d for p in parts for d in p.decisions], lambda x: x.decision, lambda x: x.evidence_segment_ids),
        open_proposals=_dedupe([d for p in parts for d in p.open_proposals], lambda x: x.proposal,
                               lambda x: x.evidence_segment_ids),
        action_items=_dedupe([d for p in parts for d in p.action_items], lambda x: x.task, lambda x: x.evidence_segment_ids),
    )


async def document(ctx: "JobContext") -> None:
    """Stage function: write 09_lm2_raw.json (without the scratchpad)."""
    s = ctx.settings
    refined = ctx.read("refined_transcript.json")
    lines = transcript_lines(refined, with_speaker=True)
    system = load_prompt("lm2_document")
    record_schema = LM2Output if s.LM2_SCRATCHPAD else LM2Record
    if not s.LM2_SCRATCHPAD:
        system += NO_SCRATCHPAD_NOTE

    async def on_retry(msg: str) -> None:
        await ctx.progress(0.5, f"Writing the meeting record: {msg}")

    n_segments = len(refined)
    # Without the scratchpad a short meeting's record fits in 4096 tokens; a smaller cap also means a smaller
    # num_ctx, which Ollama allocates and processes faster.
    lm2_max_tokens = 4096 if n_segments <= 150 and not s.LM2_SCRATCHPAD else 8192
    call = lambda user, schema=record_schema, mt=lm2_max_tokens: ctx.llm.json_call(  # noqa: E731
        s.LM2_MODEL, system, user, schema, s.LLM_MAX_RETRIES, max_tokens=mt, job_id=ctx.job_id,
        on_retry=on_retry,
    )
    if s.LM1_UNLOAD_BEFORE_LM2 and s.LM1_MODEL != s.LM2_MODEL and hasattr(ctx.llm, "unload"):
        await ctx.llm.unload(s.LM1_MODEL)  # LM1 is done; don't keep both models in RAM
    try:
        budget = s.LM2_MAX_INPUT_TOKENS - estimate_tokens(system) - 2000
        if estimate_tokens("\n".join(lines)) <= budget:
            await ctx.progress(0.1, f"Writing the meeting record with {s.LM2_MODEL}")
            out = await call(user_message(lines, record_schema))
            mode = "single"
        else:
            chunks = overlapping_chunks(lines, budget)
            parts = []
            for i, chunk in enumerate(chunks):
                await ctx.progress(i / (len(chunks) + 1), f"Writing the meeting record (part {i + 1} of {len(chunks)})")
                parts.append(await call(user_message(chunk, record_schema)))
            out = merge_outputs(parts)
            out.summary = await pick_summary(ctx, call, out)
            mode = f"chunked:{len(chunks)}"
    except (InvalidModelOutput, LLMUnavailable) as e:
        raise PipelineError("E_LM2_FAILED", Stage.DOCUMENTING, str(e)) from e
    except PipelineError:
        raise
    except Exception as e:  # noqa: BLE001 - connection errors etc.
        raise PipelineError("E_LM2_FAILED", Stage.DOCUMENTING, repr(e)) from e
    data = out.model_dump(mode="json", exclude={"scratchpad"})
    # Fallback: cross-populate if one of minutes or summary is empty
    if not data.get("minutes") and data.get("summary"):
        data["minutes"] = [{
            "topic": "General Discussion",
            "points": [s for s in data["summary"] if s.get("evidence_segment_ids")],
        }]
    elif not data.get("summary") and data.get("minutes"):
        all_pts = [p for t in data["minutes"] for p in t.get("points", []) if p.get("evidence_segment_ids")]
        data["summary"] = all_pts[:5]
    data["mode"] = mode
    ctx.write(ctx.output_name(Stage.DOCUMENTING), data)


async def pick_summary(ctx: "JobContext", call, merged: LM2Output) -> list:
    """Chunked mode: a small LM2 call that may only reuse existing cited sentences (max 5)."""
    cands = merged.summary
    if len(cands) <= 5:
        return cands
    listing = "\n".join(f"{i}. {c.text}" for i, c in enumerate(cands))
    user = (
        "These summary sentences were written for parts of one meeting. Choose at most 5 that together "
        "summarise the whole meeting without repeating each other. Return their numbers as "
        '{"keep": [..]} and nothing else.\n\n' + listing
    )
    try:
        pick: SummaryPick = await call(user, SummaryPick)
        chosen = [cands[i] for i in dict.fromkeys(pick.keep) if 0 <= i < len(cands)]
        return chosen or cands[:5]
    except Exception:  # noqa: BLE001 - fall back to the first five
        return cands[:5]
