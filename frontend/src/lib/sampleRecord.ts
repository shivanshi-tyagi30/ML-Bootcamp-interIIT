import type { Edit, MeetingRecord, Segment, Word } from "./types";

// A hand-written record that exercises the trap cases from plan section 14:
// self-correction, proposal vs decision, unassigned task, preserved negations
// and numbers, blocked edits, and a disputed name that forces "Unspecified".

type Line = [id: string, start: number, end: number, speaker: string, text: string];

const LINES: Line[] = [
  ["S001", 0.0, 6.2, "Speaker 1", "Okay, let's get started. Thanks everyone for joining the weekly sync."],
  ["S002", 6.2, 11.0, "Speaker 1", "Hi all, this is Priya from the data team, I'll take notes today."],
  ["S003", 11.0, 17.5, "Speaker 2", "Hey, Arjun here. Quick update on the release, we can't push the cuda kernels until the sea eye pipeline passes."],
  ["S004", 17.5, 23.0, "Speaker 2", "Right now the cooper netties job is failing on the GPU nodes."],
  ["S005", 23.0, 29.4, "Speaker 1", "Okay. And how did the latency look after the caching change?"],
  ["S006", 29.4, 35.0, "Speaker 2", "Latency went from fifteen to fifty milliseconds, so it actually got worse."],
  ["S007", 35.0, 41.0, "Speaker 3", "That's odd. Maybe we should move the vector store to Postgres."],
  ["S008", 41.0, 44.0, "Speaker 2", "Yeah, could be cheaper."],
  ["S009", 44.0, 51.0, "Speaker 3", "Ravi said he'll maybe look at the rag eval numbers this week."],
  ["S010", 51.0, 58.0, "Speaker 1", "On the launch, I think we've waited long enough. Okay, final call: we ship v2 without the reranker."],
  ["S011", 58.0, 60.0, "Speaker 2", "Agreed."],
  ["S012", 60.0, 62.5, "Speaker 3", "Agreed, makes sense."],
  ["S013", 62.5, 68.0, "Speaker 1", "We will not migrate the database this quarter, that's too risky before launch."],
  ["S014", 68.0, 71.5, "Speaker 2", "Fine by me."],
  ["S015", 71.5, 78.0, "Speaker 1", "I'll update the dashboards by Friday... no, Monday."],
  ["S016", 78.0, 84.0, "Speaker 2", "Someone needs to clean up the test data before we ship."],
  ["S017", 84.0, 90.5, "Speaker 3", "Arjun, can you fix the flaky integration test by Wednesday?"],
  ["S018", 90.5, 93.0, "Speaker 2", "Sure, I'll do it."],
  ["S019", 93.0, 99.0, "Speaker 3", "It would be great to have better docs someday."],
  ["S020", 99.0, 105.0, "Speaker 1", "Meera will review the key rotation by Thursday."],
  ["S021", 105.0, 110.0, "Speaker 1", "Okay, that's everything. Thanks all."],
];

// segment id -> word -> [alternative heard by the second model]
const DISPUTED: Record<string, Record<string, string>> = {
  S004: { GPU: "CPU" },
  S009: { Ravi: "Robbie" },
  S020: { Meera: "Mira" },
};

function words(id: string, text: string, start: number, end: number): Word[] {
  const toks = text.split(/\s+/);
  const step = (end - start) / toks.length;
  return toks.map((w, i) => {
    const bare = w.replace(/[^\w']/g, "");
    const alt = DISPUTED[id]?.[bare];
    return {
      w,
      start: +(start + i * step).toFixed(2),
      end: +(start + (i + 1) * step).toFixed(2),
      conf: alt ? 0.41 : 0.93,
      ...(alt ? { disputed: true, alt } : {}),
    };
  });
}

const raw: Segment[] = LINES.map(([id, start, end, speaker, text]) => ({
  id,
  start,
  end,
  speaker,
  text,
  words: words(id, text, start, end),
}));

const accepted: Edit[] = [
  { segment_id: "S003", original: "cuda", replacement: "CUDA", category: "acronym", rationale: "GPU programming term", confidence: 0.95 },
  { segment_id: "S003", original: "sea eye", replacement: "CI", category: "acronym", rationale: "'pipeline passes' implies continuous integration", confidence: 0.9 },
  { segment_id: "S004", original: "cooper netties", replacement: "Kubernetes", category: "technical_term", rationale: "Container orchestration; 'job' and 'nodes' fit", confidence: 0.88 },
  { segment_id: "S009", original: "rag eval", replacement: "RAG eval", category: "acronym", rationale: "Retrieval-augmented generation evaluation", confidence: 0.85 },
];

const applyEdits = (seg: Segment): Segment => {
  let text = seg.text;
  for (const e of accepted.filter((e) => e.segment_id === seg.id)) text = text.replace(e.original, e.replacement);
  return { ...seg, text, words: undefined };
};

export const SAMPLE_RECORD: MeetingRecord = {
  meta: {
    job_id: "sample",
    source_file: "weekly_sync.mp3",
    duration_s: 110,
    models: {
      stt: "faster-whisper large-v3",
      stt_check: "nvidia/parakeet-tdt-0.6b-v2",
      lm1: "Qwen3-8B-Instruct",
      lm2: "gemma-3-27b-it",
    },
    generated_at: "2026-10-06T10:30:00Z",
  },
  raw_transcript: raw,
  refined_transcript: raw.map(applyEdits),
  refinement: {
    accepted,
    rejected: [
      { segment_id: "S006", original: "fifteen", replacement: "fifty", category: "homophone", rationale: "Possibly misheard repeat", confidence: 0.72, reject_reason: "number_changed" },
      { segment_id: "S008", original: "cheaper", replacement: "Chroma", category: "product", rationale: "Vector database mentioned nearby", confidence: 0.71, reject_reason: "not_sound_alike" },
    ],
  },
  summary: [
    { text: "The team agreed to ship v2 without the reranker and not to migrate the database this quarter.", evidence_segment_ids: ["S010", "S011", "S012", "S013", "S014"] },
    { text: "The CUDA kernels are blocked until the CI pipeline passes, and the Kubernetes job is failing on the GPU nodes.", evidence_segment_ids: ["S003", "S004"] },
    { text: "Latency went from fifteen to fifty milliseconds after the caching change.", evidence_segment_ids: ["S005", "S006"] },
    { text: "Moving the vector store to Postgres was suggested but not agreed.", evidence_segment_ids: ["S007", "S008"] },
    { text: "Follow-ups cover the dashboards, the test data, a flaky integration test and the key rotation.", evidence_segment_ids: ["S015", "S016", "S017", "S018", "S020"] },
  ],
  minutes: [
    {
      topic: "Release blockers",
      points: [
        { text: "The CUDA kernels can't be pushed until the CI pipeline passes.", evidence_segment_ids: ["S003"] },
        { text: "The Kubernetes job is failing on the GPU nodes.", evidence_segment_ids: ["S004"] },
      ],
    },
    {
      topic: "Performance",
      points: [
        { text: "Latency went from fifteen to fifty milliseconds after the caching change, so it got worse.", evidence_segment_ids: ["S005", "S006"] },
        { text: "Moving the vector store to Postgres was suggested as possibly cheaper; no agreement was reached.", evidence_segment_ids: ["S007", "S008"] },
        { text: "Ravi may look at the RAG eval numbers this week.", evidence_segment_ids: ["S009"] },
      ],
    },
    {
      topic: "Launch plan",
      points: [
        { text: "Final call: v2 ships without the reranker; the others agreed.", evidence_segment_ids: ["S010", "S011", "S012"] },
        { text: "The database will not be migrated this quarter because it is too risky before launch.", evidence_segment_ids: ["S013", "S014"] },
      ],
    },
    {
      topic: "Follow-ups",
      points: [
        { text: "Dashboards will be updated by Monday (first said Friday, then corrected).", evidence_segment_ids: ["S015"] },
        { text: "The test data needs cleaning up before shipping; nobody took it on.", evidence_segment_ids: ["S016"] },
        { text: "Arjun agreed to fix the flaky integration test by Wednesday.", evidence_segment_ids: ["S017", "S018"] },
        { text: "The key rotation will be reviewed by Thursday.", evidence_segment_ids: ["S020"] },
      ],
    },
  ],
  decisions: [
    { id: "D1", decision: "Ship v2 without the reranker", agreement_evidence: "Agreed.", evidence_segment_ids: ["S010", "S011", "S012"] },
    { id: "D2", decision: "Do not migrate the database this quarter", agreement_evidence: "Fine by me.", evidence_segment_ids: ["S013", "S014"] },
  ],
  open_proposals: [{ proposal: "Move the vector store to Postgres", evidence_segment_ids: ["S007", "S008"] }],
  action_items: [
    {
      id: "T1",
      task: "Update the dashboards",
      owner: "Priya",
      owner_evidence: { segment_id: "S002", exact_words: "Priya" },
      deadline: "Monday",
      deadline_evidence: { segment_id: "S015", exact_words: "Monday" },
      evidence_quote: "I'll update the dashboards by Friday... no, Monday.",
      evidence_segment_ids: ["S015"],
      verifier_flags: [],
    },
    {
      id: "T2",
      task: "Clean up the test data",
      owner: "Unspecified",
      owner_evidence: null,
      deadline: "Unspecified",
      deadline_evidence: null,
      evidence_quote: "Someone needs to clean up the test data before we ship.",
      evidence_segment_ids: ["S016"],
      verifier_flags: [],
    },
    {
      id: "T3",
      task: "Fix the flaky integration test",
      owner: "Arjun",
      owner_evidence: { segment_id: "S017", exact_words: "Arjun" },
      deadline: "by Wednesday",
      deadline_evidence: { segment_id: "S017", exact_words: "by Wednesday" },
      evidence_quote: "Arjun, can you fix the flaky integration test by Wednesday?",
      evidence_segment_ids: ["S017", "S018"],
      verifier_flags: [],
    },
    {
      id: "T4",
      task: "Review the key rotation",
      owner: "Unspecified",
      owner_evidence: { segment_id: "S020", exact_words: "Meera" },
      deadline: "by Thursday",
      deadline_evidence: { segment_id: "S020", exact_words: "by Thursday" },
      evidence_quote: "Meera will review the key rotation by Thursday.",
      evidence_segment_ids: ["S020"],
      verifier_flags: ["owner_downgraded", "audio_unclear"],
    },
  ],
  fidelity: {
    numbers_preserved: "3/3",
    negations_preserved: "3/3",
    edits_accepted: 4,
    edits_rejected: 2,
    disputed_words: 3,
    items_downgraded_by_verifier: 1,
    transcript_coverage_pct: 81,
  },
};
