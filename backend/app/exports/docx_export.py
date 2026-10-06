"""Word record export with python-docx (spec Section 10.14)."""

from __future__ import annotations

from pathlib import Path

from app.exports.txt import hms
from app.models.record import MeetingRecord


def _cite(ids: list[str]) -> str:
    """'[S001] [S002]'."""
    return " ".join(f"[{i}]" for i in ids)


def write_docx(record: MeetingRecord, path: Path) -> None:
    """Write the record to a .docx file with the same sections as the Markdown export."""
    from docx import Document

    r = record
    doc = Document()
    doc.add_heading(r.meta.title, level=0)
    meta = doc.add_paragraph()
    meta.add_run(f"Date: {r.meta.generated_at:%d %b %Y}    Duration: {hms(r.meta.duration_s)}\n")
    meta.add_run(f"Source file: {r.meta.source_file}\n")
    meta.add_run("Models: " + "; ".join(f"{k}: {v}" for k, v in r.meta.models.items()))
    if r.meta.warnings:
        meta.add_run("\nWarnings: " + ", ".join(r.meta.warnings))

    doc.add_heading("Summary", level=1)
    for s in r.summary or []:
        doc.add_paragraph(f"{s.text} {_cite(s.evidence_segment_ids)}")
    if not r.summary:
        doc.add_paragraph("No summary was produced.")

    doc.add_heading("Minutes", level=1)
    for t in r.minutes:
        doc.add_heading(t.topic, level=2)
        for p in t.points:
            doc.add_paragraph(f"{p.text} {_cite(p.evidence_segment_ids)}", style="List Bullet")
    if not r.minutes:
        doc.add_paragraph("No minutes were produced.")

    doc.add_heading("Key decisions", level=1)
    for d in r.decisions:
        doc.add_paragraph(
            f"{d.id} {d.decision} (agreement: “{d.agreement_evidence}”) {_cite(d.evidence_segment_ids)}",
            style="List Bullet",
        )
    if not r.decisions:
        doc.add_paragraph("No decisions were reached.")

    doc.add_heading("Not agreed (open proposals)", level=1)
    for p in r.open_proposals:
        extra = " (no clear agreement)" if p.demoted_from_decision else ""
        doc.add_paragraph(f"{p.proposal}{extra} {_cite(p.evidence_segment_ids)}", style="List Bullet")
    if not r.open_proposals:
        doc.add_paragraph("None.")

    doc.add_heading("Action items", level=1)
    if r.action_items:
        table = doc.add_table(rows=1, cols=5)
        table.style = "Table Grid"
        for cell, h in zip(table.rows[0].cells, ["#", "Task", "Owner", "Deadline", "Evidence"]):
            cell.text = h
        for a in r.action_items:
            row = table.add_row().cells
            for cell, v in zip(row, [a.id, a.task, a.owner, a.deadline, ", ".join(a.evidence_segment_ids)]):
                cell.text = v
    else:
        doc.add_paragraph("No action items were identified.")

    f = r.fidelity
    doc.add_heading("Fidelity", level=1)
    for line in [
        f"Numbers preserved: {f.numbers_preserved}",
        f"Negations preserved: {f.negations_preserved}",
        f"Edits accepted / rejected: {f.edits_accepted} / {f.edits_rejected}",
        f"Disputed words: {f.disputed_words}",
        f"Items downgraded by verifier: {f.items_downgraded_by_verifier}",
        f"Sentences removed by verifier: {f.sentences_removed_by_verifier}",
        f"Transcript coverage: {f.transcript_coverage_pct}%",
    ]:
        doc.add_paragraph(line, style="List Bullet")
    doc.save(str(path))
