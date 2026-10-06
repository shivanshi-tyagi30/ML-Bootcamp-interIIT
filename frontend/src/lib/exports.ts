import { fmtTime } from "./format";
import type { MeetingRecord, PartialRecord, Segment } from "./types";

// Client-side exports, used in mock mode and as a fallback. The backend's
// Jinja2 templates should produce the same layout so both stay in sync.

export function transcriptTxt(segments: Segment[]): string {
  return segments
    .map((s) => `[${s.id}] ${fmtTime(s.start)}-${fmtTime(s.end)}${s.speaker ? ` (${s.speaker})` : ""}\n${s.text}`)
    .join("\n\n");
}

const cite = (ids: string[]) => (ids.length ? ` [${ids.join(", ")}]` : "");

export function recordMarkdown(r: MeetingRecord): string {
  const out: string[] = [];
  out.push(`# Meeting record: ${r.meta.source_file ?? "recording"}`, "");
  if (r.meta.duration_s != null) out.push(`- Duration: ${fmtTime(r.meta.duration_s)}`);
  const m = r.meta.models ?? {};
  out.push(`- Models: STT ${m.stt ?? "?"}${m.stt_check ? ` (+ ${m.stt_check})` : ""}; LM1 ${m.lm1 ?? "?"}; LM2 ${m.lm2 ?? "?"}`);
  if (r.meta.generated_at) out.push(`- Generated: ${r.meta.generated_at}`);
  out.push("", "## Summary", "");
  for (const s of r.summary) out.push(`${s.text}${cite(s.evidence_segment_ids)}`, "");
  out.push("## Minutes", "");
  for (const t of r.minutes) {
    out.push(`### ${t.topic}`, "");
    for (const p of t.points) out.push(`- ${p.text}${cite(p.evidence_segment_ids)}`);
    out.push("");
  }
  out.push("## Decisions", "");
  if (!r.decisions.length) out.push("No decisions were reached.");
  for (const d of r.decisions)
    out.push(`- **${d.id}** ${d.decision} (agreement: "${d.agreement_evidence}")${cite(d.evidence_segment_ids)}`);
  out.push("", "## Action items", "");
  if (!r.action_items.length) out.push("No action items were identified.");
  else {
    out.push("| ID | Task | Owner | Deadline | Evidence |", "|---|---|---|---|---|");
    for (const a of r.action_items)
      out.push(`| ${a.id} | ${a.task} | ${a.owner} | ${a.deadline} | ${a.evidence_segment_ids.join(", ")} |`);
  }
  out.push("", "## Open proposals (not agreed)", "");
  if (!r.open_proposals.length) out.push("None.");
  for (const p of r.open_proposals) out.push(`- ${p.proposal}${cite(p.evidence_segment_ids)}`);
  out.push("");
  return out.join("\n");
}

export function hasRecord(r: PartialRecord): r is MeetingRecord {
  return Array.isArray(r.summary) && Array.isArray(r.action_items) && Array.isArray(r.decisions);
}

export function downloadText(filename: string, text: string, mime = "text/plain") {
  const url = URL.createObjectURL(new Blob([text], { type: `${mime};charset=utf-8` }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
