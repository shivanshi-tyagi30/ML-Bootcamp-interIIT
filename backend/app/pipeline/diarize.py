"""DIARIZING: speaker labels with ECAPA-TDNN or pyannote (spec Section 10.6)."""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import TYPE_CHECKING, Any, Protocol

from app.config import Settings
from app.core.stages import Stage
from app.pipeline.recheck import flatten_words

if TYPE_CHECKING:
    from app.pipeline.runner import JobContext

log = logging.getLogger(__name__)

PIPELINE_V3 = "pyannote/speaker-diarization-3.1"
PIPELINE_V4 = "pyannote/speaker-diarization-community-1"
DEFAULT_ECAPA_MODEL = "speechbrain/spkrec-ecapa-voxceleb"


def pipeline_id() -> str:
    """community-1 on pyannote.audio 4+, 3.1 on older versions."""
    try:
        import pyannote.audio

        major = int(str(getattr(pyannote.audio, "__version__", "3")).split(".")[0])
    except (ImportError, ValueError):
        major = 3
    return PIPELINE_V4 if major >= 4 else PIPELINE_V3


def friendly_reason(e: Exception) -> str:
    """Short, actionable reason for a failed diarization."""
    msg = str(e)
    if "403" in msg or "gated" in msg.lower() or "401" in msg:
        return (f"Hugging Face access not granted: accept the terms on huggingface.co/{pipeline_id()} "
                "and huggingface.co/pyannote/segmentation-3.0, and check HF_TOKEN")
    return msg[:160] or repr(e)


class Diarizer(Protocol):
    """Speaker diarization interface; tests provide a fake."""

    name: str

    def run(self, path: str) -> list[dict[str, Any]]:
        """Turns [{start, end, label}] with raw labels."""
        ...


class PyannoteDiarizer:
    """pyannote speaker-diarization-3.1, loaded on first use."""

    def __init__(self, token: str) -> None:
        """Import pyannote now (raises ImportError if missing)."""
        import pyannote.audio  # noqa: F401

        self.name = pipeline_id()
        self.token = token
        self._pipe: Any = None
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        """Whether the pipeline is in memory."""
        return self._pipe is not None

    def run(self, path: str) -> list[dict[str, Any]]:
        """Diarize a WAV file."""
        from pyannote.audio import Pipeline

        with self._lock:
            if self._pipe is None:
                try:
                    self._pipe = Pipeline.from_pretrained(self.name, token=self.token)
                except TypeError:  # pyannote 3 used use_auth_token
                    self._pipe = Pipeline.from_pretrained(self.name, use_auth_token=self.token)
                if self._pipe is None:
                    raise PermissionError("403 gated model")
                try:
                    import torch

                    if torch.cuda.is_available():
                        self._pipe.to(torch.device("cuda"))
                except ImportError:
                    pass
        result = self._pipe(path)
        annotation = getattr(result, "speaker_diarization", result)
        return [
            {"start": float(t.start), "end": float(t.end), "label": str(label)}
            for t, _, label in annotation.itertracks(yield_label=True)
        ]


class EcapaDiarizer:
    """SpeechBrain ECAPA-TDNN speaker identification, loaded on first use."""

    def __init__(self, model_source: str = DEFAULT_ECAPA_MODEL) -> None:
        """Initialize the ECAPA-TDNN diarizer."""
        from app.pipeline.ecapa_diarize import get_ecapa_identifier

        self.name = model_source
        self.identifier = get_ecapa_identifier(model_source=model_source)

    @property
    def loaded(self) -> bool:
        """Whether the model is already in memory."""
        return self.identifier.loaded

    def run(self, path: str, whisper_segments: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        """Extract speaker turns by clustering embeddings for speech bursts in the audio."""
        if not whisper_segments:
            return []

        words = flatten_words(whisper_segments)
        if not words:
            return []

        bursts: list[list[dict[str, Any]]] = []
        cur_burst: list[dict[str, Any]] = [words[0]]
        for i in range(1, len(words)):
            gap = words[i]["start"] - words[i - 1]["end"]
            if gap >= 0.42:  # natural conversational turn boundary
                bursts.append(cur_burst)
                cur_burst = []
            cur_burst.append(words[i])
        if cur_burst:
            bursts.append(cur_burst)

        speaker_labels = self.identifier.identify_speakers(path, bursts)
        turns = []
        for b, spk in zip(bursts, speaker_labels):
            turns.append({
                "start": round(float(b[0]["start"]), 3),
                "end": round(float(b[-1]["end"]), 3),
                "label": spk,
            })
        return turns


_default_pyannote: PyannoteDiarizer | None = None
_default_ecapa: EcapaDiarizer | None = None
_default: Any = None


def default_pyannote_diarizer(settings: Settings) -> PyannoteDiarizer:
    """Process-wide pyannote singleton (raises ImportError without pyannote)."""
    global _default_pyannote, _default
    if _default_pyannote is None:
        _default_pyannote = PyannoteDiarizer(settings.HF_TOKEN)
        _default = _default_pyannote
    return _default_pyannote


def default_ecapa_diarizer(settings: Settings) -> EcapaDiarizer:
    """Process-wide ECAPA-TDNN diarizer singleton."""
    global _default_ecapa, _default
    if _default_ecapa is None:
        _default_ecapa = EcapaDiarizer(getattr(settings, "ECAPA_MODEL", DEFAULT_ECAPA_MODEL))
        _default = _default_ecapa
    return _default_ecapa


def relabel(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rename raw labels to 'Speaker 1', 'Speaker 2'... in order of first appearance."""
    names: dict[str, str] = {}
    out = []
    for t in sorted(turns, key=lambda x: x["start"]):
        lbl = t.get("label") or t.get("speaker", "Speaker 1")
        names.setdefault(lbl, f"Speaker {len(names) + 1}")
        out.append({"start": t["start"], "end": t["end"], "speaker": names[lbl]})
    return out


async def diarize(ctx: "JobContext") -> None:
    """Stage function: write 05_diarization.json (never fails; skips when off or unavailable)."""
    out_name = ctx.output_name(Stage.DIARIZING)
    s = ctx.settings

    if ctx.services.diarizer is None and not s.DIARIZATION_ENABLED:
        ctx.write(out_name, {"skipped": True, "reason": "turned off (DIARIZATION_ENABLED=false)", "turns": []})
        return

    # If tests or custom pipeline injected a diarizer
    if ctx.services.diarizer is not None:
        try:
            d = ctx.services.diarizer
            raw_turns = await asyncio.to_thread(d.run, str(ctx.wav_path))
            turns = relabel(raw_turns)
            ctx.write(out_name, {"skipped": False, "model": getattr(d, "name", "custom"), "turns": turns})
            return
        except Exception as e:
            ctx.log.warning("custom diarizer failed: %r", e)
            ctx.write(out_name, {"skipped": True, "reason": friendly_reason(e), "turns": []})
            return

    backend = getattr(s, "DIARIZATION_BACKEND", "auto").lower()
    use_pyannote = (backend == "pyannote") or (backend == "auto" and bool(s.HF_TOKEN))

    if use_pyannote:
        if not s.HF_TOKEN:
            ctx.write(out_name, {"skipped": True, "reason": "HF_TOKEN is not set for pyannote", "turns": []})
            return
        try:
            d = default_pyannote_diarizer(s)
            if not getattr(d, "loaded", True):
                await ctx.progress(0.0, "Loading pyannote speaker model (first run downloads it)")
            turns = relabel(await asyncio.to_thread(d.run, str(ctx.wav_path)))
            ctx.write(out_name, {"skipped": False, "model": d.name, "turns": turns})
            return
        except ImportError:
            ctx.log.warning("pyannote.audio is not installed; falling back to ECAPA-TDNN")
        except Exception as e:
            ctx.log.warning("pyannote diarization failed: %r; falling back to ECAPA-TDNN", e)

    # Run ECAPA-TDNN speaker identification
    try:
        whisper_data = ctx.read(ctx.output_name(Stage.TRANSCRIBING)) or {}
        whisper_segments = whisper_data.get("segments", [])
        ecapa = default_ecapa_diarizer(s)
        if not getattr(ecapa, "loaded", True):
            await ctx.progress(0.0, "Loading ECAPA-TDNN speaker model (first run downloads it)")
        raw_turns = await asyncio.to_thread(ecapa.run, str(ctx.wav_path), whisper_segments)
        turns = relabel(raw_turns)
        ctx.write(out_name, {"skipped": False, "model": ecapa.name, "turns": turns})
    except ImportError:
        ctx.write(out_name, {"skipped": True, "reason": "speechbrain is not installed", "turns": []})
    except Exception as e:  # noqa: BLE001 - this stage never fails the job
        ctx.log.warning("ECAPA-TDNN diarization skipped: %r", e)
        ctx.write(out_name, {"skipped": True, "reason": friendly_reason(e), "turns": []})
