// When a hand edit corrects part of a speaker's name ("Kyagi" -> "Tyagi"), rename that speaker
// everywhere: transcript labels, task owners and the record's own text ("Shivanshi Kyagi announced...").
import type { PartialRecord, Segment } from "./types";

/** The word itself, without surrounding punctuation ("Kyagi," -> "Kyagi"). */
const bare = (w: string) => w.replace(/^[^\p{L}\p{N}']+|[^\p{L}\p{N}']+$/gu, "");

const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** Whole-word replacement (letters on either side are not a match: "Ali" does not touch "Alice"). */
const replaceWord = (text: string, from: string, to: string) =>
  text.replace(new RegExp(`(?<![\\p{L}\\p{N}])${escape(from)}(?![\\p{L}\\p{N}])`, "gu"), to);

/** Speaker names (not "Speaker N" labels) that contain `word` as one of their words. */
function namesWith(record: PartialRecord, word: string): string[] {
  const all = new Set<string>();
  for (const s of [...(record.raw_transcript ?? []), ...(record.refined_transcript ?? [])]) {
    if (s.speaker && !/^Speaker \d+$/.test(s.speaker)) all.add(s.speaker);
  }
  const low = word.toLowerCase();
  return [...all].filter((n) => n.split(/\s+/).some((p) => bare(p).toLowerCase() === low));
}

/**
 * Record with every speaker name that contains `oldWord` renamed so that word reads `newWord`.
 * Unchanged when the edited word is not part of a speaker name, or the new text is not a single word.
 */
export function renameSpeakerWord(record: PartialRecord, oldWord: string, newWord: string): PartialRecord {
  const from = bare(oldWord);
  const to = bare(newWord);
  if (!from || !to || from === to || /\s/.test(newWord.trim())) return record;
  const names = namesWith(record, from);
  if (!names.length) return record;

  const rename = new Map(names.map((n) => [n, replaceWord(n, from, to)]));
  const fixText = (t: string) => replaceWord(t, from, to);
  const fixSegs = (list?: Segment[]) =>
    list?.map((s) => (s.speaker && rename.has(s.speaker) ? { ...s, speaker: rename.get(s.speaker) } : s));

  return {
    ...record,
    raw_transcript: fixSegs(record.raw_transcript),
    refined_transcript: fixSegs(record.refined_transcript),
    summary: record.summary?.map((x) => ({ ...x, text: fixText(x.text) })),
    minutes: record.minutes?.map((t) => ({
      ...t,
      topic: fixText(t.topic),
      points: t.points.map((p) => ({ ...p, text: fixText(p.text) })),
    })),
    decisions: record.decisions?.map((d) => ({ ...d, decision: fixText(d.decision) })),
    open_proposals: record.open_proposals?.map((p) => ({ ...p, proposal: fixText(p.proposal) })),
    action_items: record.action_items?.map((a) => ({
      ...a,
      task: fixText(a.task),
      owner: rename.get(a.owner) ?? fixText(a.owner),
    })),
  };
}

/** The word at `wordIdx` of a segment's text, as the transcript pane counts words. */
export function wordAt(seg: Segment | undefined, wordIdx: number): string {
  if (!seg) return "";
  return seg.text.split(/\s+/).filter(Boolean)[wordIdx] ?? "";
}

/** Pairs of words that changed when a whole line was retyped with the same number of words. */
export function changedWords(oldText: string, newText: string): [string, string][] {
  const a = oldText.trim().split(/\s+/);
  const b = newText.trim().split(/\s+/);
  if (a.length !== b.length) return [];
  return a.map((w, i) => [w, b[i]] as [string, string]).filter(([x, y]) => bare(x) !== bare(y));
}
