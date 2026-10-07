"""REFINING: LM1 pass B proposes terminology edits window by window (spec Section 10.9)."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from app.core.errors import PipelineError
from app.core.stages import Stage
from app.llm.client import LLMUnavailable, load_prompt
from app.llm.json_repair import InvalidModelOutput
from app.models.llm_io import LM1Output
from app.pipeline.lm1_vocabulary import vocabulary_json

if TYPE_CHECKING:
    from app.pipeline.runner import JobContext

# Edits are short; capping the output keeps a slow CPU model from rambling.
MAX_EDIT_TOKENS_FULL = 1536
MAX_EDIT_TOKENS_SHORT = 1024

MARKER = re.compile(r"\[\[([^|\]]*)\|[^\]]*\]\]")


def marked_text(seg: dict[str, Any]) -> str:
    """Segment text with disputed words shown as [[word|alt]]."""
    words = seg.get("words") or []
    if not any(w.get("disputed") for w in words):
        return seg["text"]
    return " ".join(f"[[{w['w']}|{w.get('alt') or ''}]]" if w.get("disputed") else w["w"] for w in words)


def strip_markers(text: str) -> str:
    """Remove [[word|alt]] markers, keeping the word."""
    return MARKER.sub(r"\1", text)


def windows(n: int, size: int, context: int) -> list[tuple[range, range, range]]:
    """(before-context, target, after-context) index ranges covering 0..n-1."""
    out = []
    for start in range(0, n, max(1, size)):
        end = min(n, start + size)
        out.append((range(max(0, start - context), start), range(start, end), range(end, min(n, end + context))))
    return out


def window_prompt(vocab: dict[str, Any], segs: list[dict[str, Any]], win: tuple[range, range, range]) -> str:
    """User message for one window."""
    before, target, after = win
    ctx_lines = [f"[{segs[i]['id']}] {segs[i]['text']}" for i in [*before, *after]]
    tgt = [{"id": segs[i]["id"], "text": marked_text(segs[i])} for i in target]
    return (
        f"VOCABULARY\n{vocabulary_json(vocab)}\n\n"
        f"CONTEXT (read-only)\n" + ("\n".join(ctx_lines) or "(none)") + "\n\n"
        f"TARGET\n{json.dumps(tgt, ensure_ascii=False)}"
    )


async def refine(ctx: "JobContext") -> None:
    """Stage function: write 07_lm1_edits.json; invalid output after retries is E_LM1_FAILED."""
    segs = ctx.read(ctx.output_name(Stage.RAW_SAVED))
    vocab = ctx.read(ctx.output_name(Stage.VOCABULARY)) or {"domain": "unknown", "terms": []}
    system = load_prompt("lm1_refine")
    wins = windows(len(segs), ctx.settings.LM1_WINDOW_SEGMENTS, ctx.settings.LM1_CONTEXT_SEGMENTS)
    max_tokens = MAX_EDIT_TOKENS_SHORT if len(segs) <= ctx.settings.LM1_WINDOW_SEGMENTS else MAX_EDIT_TOKENS_FULL
    edits: list[dict[str, Any]] = []
    domains: list[str] = []
    for k, win in enumerate(wins):
        label = f"Refining terminology (window {k + 1} of {len(wins)})"
        await ctx.progress(k / max(1, len(wins)), label)
        target_ids = {segs[i]["id"] for i in win[1]}

        async def on_retry(msg: str, k: int = k, label: str = label) -> None:
            await ctx.progress(k / max(1, len(wins)), f"{label}: {msg}")

        try:
            out = await ctx.llm.json_call(
                ctx.settings.LM1_MODEL, system, window_prompt(vocab, segs, win), LM1Output,
                ctx.settings.LLM_MAX_RETRIES, max_tokens=max_tokens, job_id=ctx.job_id, on_retry=on_retry,
            )
        except (InvalidModelOutput, LLMUnavailable) as e:
            raise PipelineError("E_LM1_FAILED", Stage.REFINING, f"window {k + 1}: {e}") from e
        except Exception as e:  # noqa: BLE001 - connection errors etc.
            raise PipelineError("E_LM1_FAILED", Stage.REFINING, repr(e)) from e
        domains.append(out.domain_guess)
        for e in out.edits:
            if e.segment_id not in target_ids:
                continue  # LM1 may only edit TARGET segments
            d = e.model_dump()
            d["original"] = strip_markers(d["original"]).strip()
            d["replacement"] = strip_markers(d["replacement"]).strip()
            if d["original"] == d["replacement"]:
                continue
            edits.append(d)
    ctx.write(ctx.output_name(Stage.REFINING), {"windows": len(wins), "domain_guess": domains, "edits": edits})
