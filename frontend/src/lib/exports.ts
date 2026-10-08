import { fmtTime } from "./format";
import type { MeetingRecord, PartialRecord, Segment } from "./types";

// Client-side exports, used in mock mode and as a fallback. The backend's
// Jinja2 templates should produce the same layout so both stay in sync.

export function transcriptTxt(segments: Segment[]): string {
  return segments
    .map((s) => `[${fmtTime(s.start)}-${fmtTime(s.end)}]${s.speaker ? ` ${s.speaker}:` : ""}\n${s.text}`)
    .join("\n\n");
}


export function recordMarkdown(r: MeetingRecord): string {
  const out: string[] = [];
  out.push(`# ${r.meta.title ?? r.meta.source_file ?? "Meeting record"}`, "");
  if (r.meta.generated_at) out.push(`- **Date:** ${r.meta.generated_at}`);
  if (r.meta.duration_s != null) out.push(`- **Duration:** ${fmtTime(r.meta.duration_s)}`);
  if (r.meta.source_file) out.push(`- **Source file:** ${r.meta.source_file}`);
  if (r.meta.warnings?.length) out.push(`- **Warnings:** ${r.meta.warnings.join(", ")}`);
  out.push("", "## Summary", "");
  for (const s of r.summary) out.push(`${s.text}`, "");
  out.push("## Minutes", "");
  for (const t of r.minutes) {
    out.push(`### ${t.topic}`, "");
    for (const p of t.points) out.push(`- ${p.text}`);
    out.push("");
  }
  out.push("## Key decisions", "");
  if (!r.decisions.length) out.push("No decisions were reached.");
  for (const d of r.decisions)
    out.push(`- **${d.id}** ${d.decision} (agreement: "${d.agreement_evidence}")`);
  out.push("", "## Action items", "");
  if (!r.action_items.length) out.push("No action items were identified.");
  else {
    out.push("| # | Task | Owner | Deadline |", "|---|---|---|---|");
    for (const a of r.action_items)
      out.push(`| ${a.id} | ${a.task} | ${a.owner} | ${a.deadline} |`);
  }
  out.push("", "## Not agreed (open proposals)", "");
  if (!r.open_proposals.length) out.push("None.");
  for (const p of r.open_proposals) out.push(`- ${p.proposal}`);
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
