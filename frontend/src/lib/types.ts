// Mirrors the MeetingRecord JSON Schema (plan section 10) and the job API
// (docs/api-contract.md). Keep in sync with the backend's Pydantic models.

export interface Word {
  w: string;
  start: number;
  end: number;
  conf?: number;
  disputed?: boolean;
  alt?: string;
}

export interface Segment {
  id: string;
  start: number;
  end: number;
  speaker?: string;
  text: string;
  words?: Word[];
}

export interface Edit {
  segment_id: string;
  original: string;
  replacement: string;
  category: string;
  rationale?: string;
  confidence: number;
}

export interface RejectedEdit extends Edit {
  reject_reason?: string;
}

export interface CitedSentence {
  text: string;
  evidence_segment_ids: string[];
}

export interface Pointer {
  segment_id: string;
  exact_words: string;
}

export interface Decision {
  id: string;
  decision: string;
  agreement_evidence: string;
  evidence_segment_ids: string[];
}

export interface Proposal {
  proposal: string;
  evidence_segment_ids: string[];
}

export type VerifierFlag =
  | "owner_downgraded"
  | "deadline_downgraded"
  | "pointer_invalid"
  | "audio_unclear";

export interface ActionItem {
  id: string;
  task: string;
  owner: string; // copied from transcript by code, or "Unspecified"
  owner_evidence?: Pointer | null;
  deadline: string; // copied verbatim by code, or "Unspecified"
  deadline_evidence?: Pointer | null;
  evidence_quote: string;
  evidence_segment_ids: string[];
  verifier_flags?: VerifierFlag[];
}

export interface Fidelity {
  numbers_preserved?: string;
  negations_preserved?: string;
  edits_accepted?: number;
  edits_rejected?: number;
  disputed_words?: number;
  items_downgraded_by_verifier?: number;
  transcript_coverage_pct?: number;
}

export interface MeetingRecord {
  meta: {
    job_id?: string;
    source_file?: string;
    duration_s?: number;
    models?: { stt?: string; stt_check?: string; lm1?: string; lm2?: string };
    generated_at?: string;
  };
  raw_transcript: Segment[];
  refined_transcript: Segment[];
  refinement: { accepted: Edit[]; rejected: RejectedEdit[] };
  summary: CitedSentence[];
  minutes: { topic: string; points: CitedSentence[] }[];
  decisions: Decision[];
  open_proposals: Proposal[];
  action_items: ActionItem[];
  fidelity: Fidelity;
}

/** While a job runs (or after a late-stage failure) only some fields exist. */
export type PartialRecord = Partial<MeetingRecord> & Pick<MeetingRecord, "meta">;

export type Stage =
  | "uploaded"
  | "validating"
  | "normalizing"
  | "speech_check"
  | "transcribing"
  | "rechecking"
  | "refining"
  | "guarding"
  | "documenting"
  | "verifying"
  | "rendering"
  | "completed"
  | "failed";

export const PIPELINE_ORDER: Stage[] = [
  "uploaded",
  "validating",
  "normalizing",
  "speech_check",
  "transcribing",
  "rechecking",
  "refining",
  "guarding",
  "documenting",
  "verifying",
  "rendering",
  "completed",
];

export type ErrorCode =
  | "E_UNSUPPORTED_FORMAT"
  | "E_EMPTY_FILE"
  | "E_TOO_LARGE"
  | "E_UNREADABLE"
  | "E_NO_SPEECH"
  | "E_STT_FAILED"
  | "E_LM1_FAILED"
  | "E_LM2_FAILED"
  | "E_INTERNAL"
  | "E_NETWORK";

export interface JobError {
  code: ErrorCode | string;
  stage?: Stage;
  user_message: string;
}

/** One Server-Sent Event from GET /jobs/{id}/events. */
export interface JobEvent {
  stage: Stage;
  error?: JobError;
  warnings?: string[]; // e.g. "W_NON_ENGLISH"
}

/** GET /jobs/{id} */
export interface JobState {
  job_id: string;
  stage: Stage;
  error?: JobError | null;
  warnings?: string[];
  record: PartialRecord;
}

export type ExportFormat = "json" | "md" | "docx" | "txt_raw" | "txt_refined";
