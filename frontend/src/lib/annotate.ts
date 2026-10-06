import type { Edit, RejectedEdit, Segment, Word } from "./types";

export type TranscriptMode = "raw" | "refined" | "diff";

export type Piece =
  | { kind: "text"; text: string }
  | { kind: "accepted"; edit: Edit }
  | { kind: "rejected"; edit: RejectedEdit }
  | { kind: "disputed"; text: string; word: Word };

interface Span {
  start: number;
  end: number;
  priority: number;
  piece: Piece;
}

/**
 * Splits a RAW segment into pieces: accepted edits, rejected edits and disputed
 * words, located by character offset. The refined transcript is, by design,
 * raw + accepted edits, so all three views are rendered from the same pieces.
 */
export function annotate(
  seg: Segment,
  accepted: Edit[],
  rejected: RejectedEdit[],
  mode: TranscriptMode,
): Piece[] {
  const text = seg.text;
  const spans: Span[] = [];

  for (const e of accepted) {
    const i = text.indexOf(e.original);
    if (i >= 0) spans.push({ start: i, end: i + e.original.length, priority: 0, piece: { kind: "accepted", edit: e } });
  }
  if (mode === "diff") {
    for (const e of rejected) {
      const i = text.indexOf(e.original);
      if (i >= 0) spans.push({ start: i, end: i + e.original.length, priority: 1, piece: { kind: "rejected", edit: e } });
    }
  }
  let cursor = 0;
  for (const w of seg.words ?? []) {
    const token = w.w.trim().replace(/^[^\w']+|[^\w']+$/g, "");
    if (!token) continue;
    const i = text.indexOf(token, cursor);
    if (i < 0) continue;
    cursor = i + token.length;
    if (w.disputed) spans.push({ start: i, end: cursor, priority: 2, piece: { kind: "disputed", text: token, word: w } });
  }

  // Keep the highest-priority span where spans overlap.
  spans.sort((a, b) => a.priority - b.priority || a.start - b.start);
  const kept: Span[] = [];
  for (const s of spans) if (!kept.some((k) => s.start < k.end && k.start < s.end)) kept.push(s);
  kept.sort((a, b) => a.start - b.start);

  const out: Piece[] = [];
  let pos = 0;
  for (const s of kept) {
    if (mode === "raw" && s.piece.kind === "accepted") continue;
    if (s.start > pos) out.push({ kind: "text", text: text.slice(pos, s.start) });
    out.push(s.piece);
    pos = s.end;
  }
  if (pos < text.length) out.push({ kind: "text", text: text.slice(pos) });
  return out;
}

export const REJECT_REASONS: Record<string, string> = {
  not_found: "the original words aren't in this segment",
  number_changed: "it would change a number",
  negation_changed: "it would add or remove a negation",
  over_rewrite: "the replacement is far longer than the original",
  not_sound_alike: "the replacement doesn't sound like what was said",
  low_confidence: "the model wasn't confident enough",
  frozen_token: "it touches a protected word (name, date, modal)",
};
