"""TRANSCRIBING: faster-whisper large-v3 with hallucination guards (spec Section 10.4)."""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from typing import TYPE_CHECKING, Any, Protocol

from app.config import Settings
from app.core.errors import PipelineError
from app.core.stages import Stage

if TYPE_CHECKING:
    from app.pipeline.runner import JobContext

log = logging.getLogger(__name__)


class STT(Protocol):
    """Speech-to-text interface; tests provide a fake."""

    name: str

    def transcribe(self, path: str, initial_prompt: str | None) -> dict[str, Any]:
        """Return {language, language_probability, segments:[{start,end,text,avg_logprob,
        no_speech_prob,compression_ratio,words:[{w,start,end,conf}]}]}."""
        ...


class WhisperSTT:
    """faster-whisper model, loaded on first use."""

    def __init__(self, settings: Settings) -> None:
        """Remember settings; the model loads lazily."""
        self.settings = settings
        self.name = f"faster-whisper {settings.WHISPER_MODEL}"
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        """Whether the model is in memory."""
        return self._model is not None

    def _load(self) -> Any:
        """Load the model once."""
        with self._lock:
            if self._model is None:
                from faster_whisper import WhisperModel

                s = self.settings
                self._model = WhisperModel(s.WHISPER_MODEL, device=s.WHISPER_DEVICE, compute_type=s.WHISPER_COMPUTE_TYPE)
            return self._model

    def transcribe(self, path: str, initial_prompt: str | None) -> dict[str, Any]:
        """Transcribe with word timestamps (spec parameters)."""
        model = self._load()
        segments, info = model.transcribe(
            path, language=None, task="transcribe", beam_size=5, temperature=0.0, word_timestamps=True,
            vad_filter=True, condition_on_previous_text=False, initial_prompt=initial_prompt,
        )
        out = []
        for s in segments:
            out.append({
                "start": s.start, "end": s.end, "text": s.text.strip(), "avg_logprob": s.avg_logprob,
                "no_speech_prob": s.no_speech_prob, "compression_ratio": s.compression_ratio,
                "words": [
                    {"w": w.word.strip(), "start": w.start, "end": w.end, "conf": round(w.probability, 4)}
                    for w in (s.words or []) if w.word.strip()
                ],
            })
        return {"language": info.language, "language_probability": info.language_probability, "segments": out}


def glossary_prompt(terms: list[str]) -> str | None:
    """'Glossary: a, b, c' capped at roughly 200 tokens, or None without terms."""
    terms = [t.strip() for t in terms if t.strip()]
    if not terms:
        return None
    prompt = "Glossary: "
    for t in terms:
        nxt = prompt + (", " if prompt != "Glossary: " else "") + t
        if len(nxt) > 800:  # ~200 tokens
            break
        prompt = nxt
    return prompt


def has_repeated_phrase(text: str, min_words: int = 4, repeats: int = 3) -> bool:
    """True if some phrase of 4+ words repeats 3+ times in a row (a common Whisper loop)."""
    toks = re.findall(r"[\w']+", text.lower())
    for n in range(min_words, len(toks) // repeats + 1):
        for i in range(len(toks) - n * repeats + 1):
            if all(toks[i : i + n] == toks[i + k * n : i + (k + 1) * n] for k in range(1, repeats)):
                return True
    return False


def apply_hallucination_guards(segments: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split segments into (kept, dropped) using the spec's three guards."""
    kept, dropped = [], []
    for s in segments:
        reason = None
        if s.get("no_speech_prob", 0) > 0.6 and s.get("avg_logprob", 0) < -1.0:
            reason = "no_speech"
        elif s.get("compression_ratio", 0) > 2.4:
            reason = "compression_ratio"
        elif has_repeated_phrase(s.get("text", "")):
            reason = "repetition"
        (dropped if reason else kept).append({**s, "drop_reason": reason} if reason else s)
    return kept, dropped


_default: WhisperSTT | None = None


def default_stt(settings: Settings) -> WhisperSTT:
    """Process-wide Whisper singleton."""
    global _default
    if _default is None:
        _default = WhisperSTT(settings)
    return _default


async def transcribe(ctx: "JobContext") -> None:
    """Stage function: write 03_whisper.json."""
    stt = ctx.services.stt or default_stt(ctx.settings)
    prompt = glossary_prompt(ctx.upload.get("glossary", []))
    try:
        result = await asyncio.to_thread(stt.transcribe, str(ctx.wav_path), prompt)
    except PipelineError:
        raise
    except Exception as e:  # noqa: BLE001 - any model failure is E_STT_FAILED
        raise PipelineError("E_STT_FAILED", Stage.TRANSCRIBING, repr(e)) from e
    kept, dropped = apply_hallucination_guards(result.get("segments", []))
    if not kept or not any(s["words"] for s in kept):
        raise PipelineError("E_NO_SPEECH", Stage.TRANSCRIBING, "no segments after guards")
    lang, prob = result.get("language", "en"), float(result.get("language_probability", 1.0))
    warnings = ["W_NON_ENGLISH"] if lang != "en" or prob < 0.5 else []
    ctx.write(ctx.output_name(Stage.TRANSCRIBING), {
        "model": getattr(stt, "name", "whisper"), "language": lang, "language_probability": prob,
        "warnings": warnings, "segments": kept, "dropped": dropped,
    })
    for w in warnings:
        ctx.add_warning(w)
