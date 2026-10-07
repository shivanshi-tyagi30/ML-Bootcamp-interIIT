"""TRANSCRIBING: faster-whisper large-v3 with hallucination guards (spec Section 10.4)."""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from typing import TYPE_CHECKING, Any, Callable, Protocol

from app.config import Settings
from app.core.errors import JobCancelled as Cancelled
from app.core.errors import PipelineError
from app.core.stages import Stage

if TYPE_CHECKING:
    from app.pipeline.runner import JobContext

log = logging.getLogger(__name__)


class STT(Protocol):
    """Speech-to-text interface; tests provide a fake."""

    name: str

    def transcribe(
        self, path: str, initial_prompt: str | None, progress: Callable[[float], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Return {language, language_probability, segments:[{start,end,text,avg_logprob,
        no_speech_prob,compression_ratio,words:[{w,start,end,conf}]}]}."""
        ...


class WhisperSTT:
    """faster-whisper model, loaded on first use."""

    def __init__(self, settings: Settings) -> None:
        """Remember settings; the model loads lazily."""
        self.settings = settings
        self.rt = settings.whisper_runtime()
        self.name = f"faster-whisper {self.rt['model']} ({self.rt['device']}, {self.rt['compute_type']})"
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

                rt = self.rt
                log.info("loading Whisper %s on %s (%s)", rt["model"], rt["device"], rt["compute_type"])
                kw = dict(device=str(rt["device"]), compute_type=str(rt["compute_type"]),
                          cpu_threads=int(rt["cpu_threads"]))
                try:  # already fetched by `python -m app.prefetch`: no network round trip
                    self._model = WhisperModel(str(rt["model"]), local_files_only=True, **kw)
                except Exception:  # noqa: BLE001 - not cached yet
                    self._model = WhisperModel(str(rt["model"]), **kw)
            return self._model

    def transcribe(
        self, path: str, initial_prompt: str | None, progress: Callable[[float], None] | None = None,
        should_stop: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Transcribe with word timestamps; beam size is 5 on GPU and 1 on CPU unless set."""
        model = self._load()
        segments, info = model.transcribe(
            path, language=None, task="transcribe", beam_size=int(self.rt["beam_size"]), temperature=0.0,
            word_timestamps=True, vad_filter=True, condition_on_previous_text=False, initial_prompt=initial_prompt,
        )
        total = float(getattr(info, "duration", 0) or 0)
        out = []
        for s in segments:  # a lazy generator: decoding happens as we iterate
            if should_stop and should_stop():
                raise Cancelled()
            if progress and total:
                progress(min(1.0, s.end / total))
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


def _mmss(sec: float) -> str:
    """m:ss."""
    t = int(max(0, sec))
    return f"{t // 60}:{t % 60:02d}"


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
    if not getattr(stt, "loaded", True):
        await ctx.progress(0.0, "Preparing the speech model")
    duration = float((ctx.read(ctx.output_name(Stage.NORMALIZING)) or {}).get("duration_s") or 0)

    def report(frac: float) -> None:
        done = f" ({_mmss(frac * duration)} of {_mmss(duration)})" if duration else ""
        ctx.progress_threadsafe(frac, f"Transcribing{done}")

    try:
        result = await asyncio.to_thread(
            stt.transcribe, str(ctx.wav_path), prompt, progress=report, should_stop=ctx.cancel_requested,
        )
    except Cancelled:
        raise
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
