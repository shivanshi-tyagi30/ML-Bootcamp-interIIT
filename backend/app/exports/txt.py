"""Plain-text transcript exports (spec Section 10.14)."""

from __future__ import annotations

from app.models.record import Segment


def hms(seconds: float) -> str:
    """HH:MM:SS."""
    t = int(max(0, seconds))
    return f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


def transcript_txt(segments: list[Segment]) -> str:
    """One line per segment: '[HH:MM:SS] Speaker 1: text' (speaker omitted if None)."""
    return "\n".join(
        f"[{hms(s.start)}] {s.speaker + ': ' if s.speaker else ''}{s.text}" for s in segments
    ) + "\n"
