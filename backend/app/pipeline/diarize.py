"""DIARIZING: optional speaker labels with pyannote (spec Section 10.6)."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any, Protocol

from app.config import Settings
from app.core.stages import Stage

if TYPE_CHECKING:
    from app.pipeline.runner import JobContext

PIPELINE_V3 = "pyannote/speaker-diarization-3.1"
PIPELINE_V4 = "pyannote/speaker-diarization-community-1"


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


_default: PyannoteDiarizer | None = None


def default_diarizer(settings: Settings) -> PyannoteDiarizer:
    """Process-wide pyannote singleton (raises ImportError without pyannote)."""
    global _default
    if _default is None:
        _default = PyannoteDiarizer(settings.HF_TOKEN)
    return _default


def relabel(turns: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rename raw labels to 'Speaker 1', 'Speaker 2'... in order of first appearance."""
    names: dict[str, str] = {}
    out = []
    for t in sorted(turns, key=lambda x: x["start"]):
        names.setdefault(t["label"], f"Speaker {len(names) + 1}")
        out.append({"start": t["start"], "end": t["end"], "speaker": names[t["label"]]})
    return out


async def diarize(ctx: "JobContext") -> None:
    """Stage function: write 05_diarization.json (never fails; skips when off or unavailable)."""
    out_name = ctx.output_name(Stage.DIARIZING)
    s = ctx.settings
    if ctx.services.diarizer is None and not s.DIARIZATION_ENABLED:
        ctx.write(out_name, {"skipped": True, "reason": "turned off (DIARIZATION_ENABLED=false)", "turns": []})
        return
    if ctx.services.diarizer is None and not s.HF_TOKEN:
        ctx.write(out_name, {"skipped": True, "reason": "HF_TOKEN is not set", "turns": []})
        return
    try:
        d = ctx.services.diarizer or default_diarizer(s)
        if not getattr(d, "loaded", True):
            await ctx.progress(0.0, "Loading the speaker model (first run downloads it)")
        turns = relabel(await asyncio.to_thread(d.run, str(ctx.wav_path)))
        ctx.write(out_name, {"skipped": False, "model": d.name, "turns": turns})
    except ImportError:
        ctx.write(out_name, {"skipped": True, "reason": "pyannote.audio is not installed", "turns": []})
    except Exception as e:  # noqa: BLE001 - this stage never fails the job
        ctx.log.warning("diarization skipped: %r", e)
        ctx.write(out_name, {"skipped": True, "reason": friendly_reason(e), "turns": []})
