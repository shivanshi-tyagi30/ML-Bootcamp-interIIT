"""VOCABULARY: LM1 pass A builds the meeting's own term list (spec Section 10.8)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from app.core.stages import Stage
from app.core.text import estimate_tokens
from app.llm.client import load_prompt
from app.models.llm_io import Vocabulary, VocabularyTerm

if TYPE_CHECKING:
    from app.pipeline.runner import JobContext

CHUNK_TOKENS = 20_000


def transcript_lines(segments: list[dict[str, Any]], with_speaker: bool = False) -> list[str]:
    """'[S001] text' (or '[S001] (Speaker 1) text') lines."""
    out = []
    for s in segments:
        who = f"({s['speaker']}) " if with_speaker and s.get("speaker") else ""
        out.append(f"[{s['id']}] {who}{s['text']}")
    return out


def chunk_lines(lines: list[str], max_tokens: int) -> list[list[str]]:
    """Split lines into chunks under `max_tokens` each, never splitting a line."""
    chunks: list[list[str]] = [[]]
    size = 0
    for ln in lines:
        t = estimate_tokens(ln) + 1
        if chunks[-1] and size + t > max_tokens:
            chunks.append([])
            size = 0
        chunks[-1].append(ln)
        size += t
    return chunks


def merge_vocabularies(parts: list[Vocabulary], glossary: list[str]) -> dict[str, Any]:
    """Merge chunk results (dedupe case-insensitively) and add user glossary terms."""
    merged: dict[str, VocabularyTerm] = {}
    for p in parts:
        for t in p.terms:
            key = t.term.strip().lower()
            if not key:
                continue
            if key in merged:
                m = merged[key]
                m.heard_as = sorted(set(m.heard_as) | set(t.heard_as))
                m.evidence_segment_ids = sorted(set(m.evidence_segment_ids) | set(t.evidence_segment_ids))
            else:
                merged[key] = t.model_copy()
    for g in glossary:
        if g.strip() and g.strip().lower() not in merged:
            merged[g.strip().lower()] = VocabularyTerm(term=g.strip(), heard_as=[], evidence_segment_ids=[])
    domain = next((p.domain for p in parts if p.domain), "unknown")
    return {"domain": domain, "terms": [t.model_dump() for t in list(merged.values())[:60 + len(glossary)]]}


async def make_room_for(ctx: "JobContext", model: str) -> None:
    """One-model-at-a-time mode (LM1_UNLOAD_BEFORE_LM2): free the other LLM before `model` runs.

    On a 16 GB laptop qwen3:8b and gemma3:12b do not fit in memory together; holding both makes the
    system swap to disk, which is far slower than reloading one model.
    """
    s = ctx.settings
    other = s.LM2_MODEL if model == s.LM1_MODEL else s.LM1_MODEL
    if s.LM1_UNLOAD_BEFORE_LM2 and other != model and hasattr(ctx.llm, "unload"):
        await ctx.llm.unload(other)


async def build_vocabulary(ctx: "JobContext") -> None:
    """Stage function: write 06_vocabulary.json; any failure yields an empty vocabulary."""
    out_name = ctx.output_name(Stage.VOCABULARY)
    glossary = ctx.upload.get("glossary", [])
    raw = ctx.read(ctx.output_name(Stage.RAW_SAVED))
    mode = ctx.settings.VOCAB_PASS
    one_window = len(raw) <= ctx.settings.LM1_WINDOW_SEGMENTS
    if mode == "never" or (mode == "auto" and one_window):
        # The refiner sees the whole meeting in one call, so a separate pass adds a full LLM call for
        # little gain. The user's expected terms and the known-terms lists still apply.
        vocab = merge_vocabularies([Vocabulary(domain="unknown", terms=[])], glossary)
        ctx.log.info("vocabulary pass skipped (short meeting, one refine window)")
        ctx.write(out_name, vocab)
        return
    await make_room_for(ctx, ctx.settings.LM1_MODEL)
    try:
        chunks = chunk_lines(transcript_lines(raw), CHUNK_TOKENS)
        system = load_prompt("lm1_vocabulary")
        parts = []
        for i, chunk in enumerate(chunks):
            await ctx.progress(i / len(chunks), f"Learning the meeting's vocabulary (part {i + 1} of {len(chunks)})")
            vocab_max_tokens = 1536 if len(chunks) == 1 else 2048
            parts.append(await ctx.llm.json_call(
                ctx.settings.LM1_MODEL, system, "TRANSCRIPT\n" + "\n".join(chunk), Vocabulary,
                min(1, ctx.settings.LLM_MAX_RETRIES), max_tokens=vocab_max_tokens, job_id=ctx.job_id,
            ))
        vocab = merge_vocabularies(parts, glossary)
    except Exception as e:  # noqa: BLE001 - this stage never fails the job
        ctx.log.warning("vocabulary pass failed (%r); continuing with an empty vocabulary", e)
        vocab = merge_vocabularies([Vocabulary(domain="unknown", terms=[])], glossary)
    ctx.write(out_name, vocab)


def vocabulary_json(vocab: dict[str, Any]) -> str:
    """Compact JSON of the vocabulary for prompts."""
    return json.dumps(vocab, ensure_ascii=False)
