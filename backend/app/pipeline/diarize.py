"""DIARIZING: optional speaker labels with pyannote (spec Section 10.6)."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any, Protocol

from app.config import Settings
from app.core.stages import Stage

if TYPE_CHECKING:
    from app.pipeline.runner import JobContext

PIPELINE_ID = "pyannote/speaker-diarization-3.1"


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

        self.name = PIPELINE_ID
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
                    self._pipe = Pipeline.from_pretrained(PIPELINE_ID, use_auth_token=self.token)
                except TypeError:  # pyannote >= 4 renamed the argument
                    self._pipe = Pipeline.from_pretrained(PIPELINE_ID, token=self.token)
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
    if ctx.services.diarizer is None and not (s.DIARIZATION_ENABLED and s.HF_TOKEN):
        ctx.write(out_name, {"skipped": True, "reason": "disabled", "turns": []})
        return
    try:
        d = ctx.services.diarizer or default_diarizer(s)
        turns = relabel(await asyncio.to_thread(d.run, str(ctx.wav_path)))
        ctx.write(out_name, {"skipped": False, "model": d.name, "turns": turns})
    except Exception as e:  # noqa: BLE001 - this stage never fails the job
        ctx.log.warning("diarization skipped: %r", e)
        ctx.write(out_name, {"skipped": True, "reason": f"error: {e!r}", "turns": []})
