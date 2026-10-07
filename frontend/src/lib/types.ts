// Mirrors the MeetingRecord JSON Schema (plan section 10) and the job API
// (docs/api-contract.md). Keep in sync with the backend's Pydantic models.

export interface Word {
  w: string;
  start: number;
  end: number;
  conf?: number;
  disputed?: boolean;
  alt?: string | null; // null: flagged for low confidence, no second model ran
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
  demoted_from_decision?: boolean;
}

export type VerifierFlag =
  | "owner_downgraded"
  | "deadline_downgraded"
  | "pointer_invalid"
  | "audio_unclear"
  | "self_assignment_unverified";

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
  sentences_removed_by_verifier?: number;
  transcript_coverage_pct?: number;
}

export interface MeetingRecord {
  meta: {
    job_id?: string;
    title?: string;
    source_file?: string;
    duration_s?: number;
    language?: string;
    language_probability?: number;
    models?: { stt?: string; stt_check?: string; diarization?: string; lm1?: string; lm2?: string };
    warnings?: string[];
    generated_at?: string;
    /** Seconds per pipeline stage (from the job's timings.json; not part of record.json). */
    timings?: Record<string, number>;
  };
  raw_transcript: Segment[];
  refined_transcript: Segment[];
  refinement: { vocabulary?: { domain?: string; terms?: { term: string }[] }; accepted: Edit[]; rejected: RejectedEdit[] };
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
  | "queued"
  | "validating"
  | "normalizing"
  | "speech_check"
  | "transcribing"
  | "rechecking"
  | "diarizing"
  | "raw_saved"
  | "vocabulary"
  | "refining"
  | "guarding"
  | "documenting"
  | "verifying"
  | "rendering"
  | "completed"
  | "failed";

export const PIPELINE_ORDER: Stage[] = [
  "uploaded",
  "queued",
  "validating",
  "normalizing",
  "speech_check",
  "transcribing",
  "rechecking",
  "diarizing",
  "raw_saved",
  "vocabulary",
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
  | "E_TOO_LONG"
  | "E_UNREADABLE"
  | "E_NO_SPEECH"
  | "E_STT_FAILED"
  | "E_LM1_FAILED"
  | "E_LM2_FAILED"
  | "E_RENDER_FAILED"
  | "E_BUSY"
  | "E_NOT_FOUND"
  | "E_JOB_RUNNING"
  | "E_CANCELLED"
  | "E_FFMPEG_MISSING"
  | "E_INTERNAL"
  | "E_NETWORK";

export interface JobError {
  code: ErrorCode | string;
  stage?: Stage;
  user_message: string;
  /** Short technical reason from the server, e.g. which model failed and why. */
  detail?: string | null;
}

/** One `progress` Server-Sent Event from GET /api/jobs/{id}/events. */
export interface JobEvent {
  stage: Stage;
  status: "queued" | "running" | "completed" | "failed";
  progress?: number; // 0..1 overall
  message?: string; // e.g. "Refining terminology (window 3 of 6)"
  raw_ready?: boolean;
  refined_ready?: boolean;
  record_ready?: boolean;
  error?: JobError;
  warnings?: string[]; // e.g. "W_NON_ENGLISH"
}

/** A row of GET /api/jobs. */
export interface JobSummary {
  id: string;
  title: string;
  source_file: string;
  status: "queued" | "running" | "completed" | "failed";
  stage: string;
  error_code?: string | null;
  error_message?: string | null;
  error_detail?: string | null;
  duration_s?: number | null;
  n_decisions: number;
  n_tasks: number;
  created_at: string;
  updated_at: string;
}

/** GET /api/jobs/{id}, normalised for the UI. */
export interface JobState {
  job_id: string;
  stage: Stage;
  error?: JobError | null;
  warnings?: string[];
  record: PartialRecord;
}

/** GET /api/health: what the backend can reach right now. */
export interface Health {
  status: string;
  ffmpeg?: boolean;
  llm?: { backend: string; host: string; reachable: boolean | null; missing: string[] };
  whisper?: { model: string; device: string; compute_type: string; beam_size: number };
  gpu?: boolean;
}

export type ExportFormat = "json" | "md" | "docx" | "txt_raw" | "txt_refined";
