"""Exports generated only from record.json (spec Section 10.14)."""

from __future__ import annotations

from pathlib import Path

from app.core.storage import write_json
from app.exports.docx_export import write_docx
from app.exports.markdown import record_markdown
from app.exports.txt import transcript_txt
from app.models.record import MeetingRecord

EXPORT_FILES = {
    "md": "record.md",
    "docx": "record.docx",
    "txt_raw": "transcript_raw.txt",
    "txt_refined": "transcript_refined.txt",
}


def render_exports(record: MeetingRecord, out_dir: Path) -> dict[str, str]:
    """Write every export for a record; returns {fmt: filename}. The manifest is written last."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / EXPORT_FILES["md"]).write_text(record_markdown(record), encoding="utf-8")
    write_docx(record, out_dir / EXPORT_FILES["docx"])
    (out_dir / EXPORT_FILES["txt_raw"]).write_text(transcript_txt(record.raw_transcript), encoding="utf-8")
    (out_dir / EXPORT_FILES["txt_refined"]).write_text(transcript_txt(record.refined_transcript), encoding="utf-8")
    write_json(out_dir / "manifest.json", {"files": EXPORT_FILES, "title": record.meta.title})
    return EXPORT_FILES
