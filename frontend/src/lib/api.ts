import { ERROR_MESSAGES, makeError } from "./errors";
import { SAMPLE_RECORD } from "./sampleRecord";
import type { ExportFormat, Health, JobError, JobEvent, JobState, JobSummary, PartialRecord, Stage } from "./types";

// Backend contract: TRACE Backend Build Spec, sections 9 and 21 (see docs/api-contract.md).

export interface CreatedJob {
  jobId: string;
  /** True when the same file was already processed; the saved result opens directly. */
  cached: boolean;
}

export interface Api {
  mock: boolean;
  createJob(file: File, opts: { title?: string; glossary?: string; force?: boolean }): Promise<CreatedJob>;
  /** Calls onEvent for every progress event until completed/failed. Returns an unsubscribe fn. */
  watch(jobId: string, onEvent: (e: JobEvent) => void): () => void;
  getJob(jobId: string): Promise<JobState>;
  listJobs(): Promise<JobSummary[]>;
  renameJob(jobId: string, title: string): Promise<JobSummary>;
  deleteJob(jobId: string): Promise<void>;
  /** Stop a queued or running job; finished steps stay saved. */
  cancelJob(jobId: string): Promise<JobSummary>;
  /** Run a failed or cancelled job again from the step that stopped. */
  retryJob(jobId: string): Promise<JobSummary>;
  /** Backend readiness (ffmpeg, Ollama and its models), or null if unreachable. */
  health(): Promise<Health | null>;
  /** Backend download URL, or null when the client should build the file itself. */
  exportUrl(jobId: string, fmt: ExportFormat): string | null;
  /** Seekable original recording served by the backend, or null. */
  audioUrl(jobId: string): string | null;
}

export class ApiError extends Error {
  constructor(public jobError: JobError) {
    super(jobError.user_message);
  }
}

const BASE = (import.meta.env.VITE_API_URL ?? "/api").replace(/\/$/, "");

async function readError(res: Response): Promise<JobError> {
  try {
    const body = await res.json();
    const err = body.detail?.code ? body.detail : body.error?.code ? body.error : body;
    if (err?.code)
      return {
        code: err.code,
        stage: err.stage,
        user_message: err.message ?? err.user_message ?? ERROR_MESSAGES.E_INTERNAL,
      };
  } catch {
    /* not JSON */
  }
  return makeError("E_INTERNAL", `Server error (${res.status}).`);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    // The cloud API key travels only with uploads (createJob), never with every request.
    res = await fetch(`${BASE}${path}`, init);
  } catch {
    throw new ApiError(makeError("E_NETWORK"));
  }
  if (!res.ok) throw new ApiError(await readError(res));
  return (res.status === 204 ? undefined : await res.json()) as T;
}

interface ServerEvent {
  stage: Stage;
  status: JobEvent["status"];
  progress?: number;
  message?: string;
  raw_ready?: boolean;
  refined_ready?: boolean;
  record_ready?: boolean;
  warnings?: string[];
  error_code?: string;
  error_message?: string;
  error_detail?: string | null;
  failed_stage?: Stage;
}

function toJobEvent(e: ServerEvent): JobEvent {
  const failed = e.status === "failed";
  return {
    stage: failed ? "failed" : e.stage,
    status: e.status,
    progress: e.progress,
    message: e.message,
    raw_ready: e.raw_ready,
    refined_ready: e.refined_ready,
    record_ready: e.record_ready,
    warnings: e.warnings,
    error: failed
      ? {
          code: e.error_code ?? "E_INTERNAL",
          stage: e.failed_stage ?? (e.stage as Stage),
          user_message: e.error_message ?? ERROR_MESSAGES.E_INTERNAL,
          detail: e.error_detail,
        }
      : undefined,
  };
}

interface ServerJobDetail {
  job: JobSummary;
  record: PartialRecord | null;
  partial: {
    raw_transcript: PartialRecord["raw_transcript"] | null;
    refined_transcript: PartialRecord["refined_transcript"] | null;
    refinement: PartialRecord["refinement"] | null;
  };
  timings?: Record<string, number> | null;
}

function toJobState(d: ServerJobDetail): JobState {
  const { job } = d;
  const base: PartialRecord = d.record ?? {
    meta: { job_id: job.id, title: job.title, source_file: job.source_file, duration_s: job.duration_s ?? undefined },
    raw_transcript: d.partial.raw_transcript ?? undefined,
    refined_transcript: d.partial.refined_transcript ?? undefined,
    refinement: d.partial.refinement ?? undefined,
  };
  const record: PartialRecord = d.timings ? { ...base, meta: { ...base.meta, timings: d.timings } } : base;
  return {
    job_id: job.id,
    stage: (job.status === "failed" ? "failed" : job.stage) as Stage,
    error:
      job.status === "failed"
        ? {
            code: job.error_code ?? "E_INTERNAL",
            user_message: job.error_message ?? ERROR_MESSAGES.E_INTERNAL,
            detail: job.error_detail,
          }
        : null,
    record,
  };
}

const httpApi: Api = {
  mock: false,

  async createJob(file, { title, glossary, force }) {
    const form = new FormData();
    form.append("file", file);
    if (title?.trim()) form.append("title", title.trim());
    if (glossary?.trim()) form.append("glossary", glossary.trim());
    const apiKey = typeof window !== "undefined" ? localStorage.getItem("trace_api_key")?.trim() : null;
    if (apiKey) form.append("api_key", apiKey);
    const body = await request<{ job_id: string; cached?: boolean }>(`/jobs${force ? "?force=true" : ""}`, {
      method: "POST",
      body: form,
    });
    return { jobId: body.job_id, cached: !!body.cached };
  },

  watch(jobId, onEvent) {
    let closed = false;
    let pollTimer: number | undefined;
    let last = "";

    const emit = (e: JobEvent) => {
      if (closed) return;
      const key = `${e.stage}|${e.status}|${e.message}|${e.progress}`;
      if (key === last) return;
      last = key;
      onEvent(e);
      if (e.status === "completed" || e.status === "failed") stop();
    };

    // Fallback if SSE drops (proxies, restarts): poll the job state.
    const poll = async () => {
      if (closed) return;
      try {
        const d = await request<ServerJobDetail>(`/jobs/${jobId}`);
        emit(
          toJobEvent({
            stage: d.job.stage as Stage,
            status: d.job.status,
            raw_ready: !!d.partial.raw_transcript,
            refined_ready: !!d.partial.refined_transcript,
            record_ready: !!d.record,
            error_code: d.job.error_code ?? undefined,
            error_message: d.job.error_message ?? undefined,
            error_detail: d.job.error_detail,
          }),
        );
      } catch {
        /* keep trying */
      }
      if (!closed) pollTimer = window.setTimeout(poll, 2000);
    };

    const es = new EventSource(`${BASE}/jobs/${jobId}/events`);
    es.addEventListener("progress", (m) => {
      try {
        emit(toJobEvent(JSON.parse((m as MessageEvent).data) as ServerEvent));
      } catch {
        /* ignore malformed event */
      }
    });
    es.onerror = () => {
      es.close();
      if (!closed && pollTimer === undefined) poll();
    };

    function stop() {
      closed = true;
      es.close();
      window.clearTimeout(pollTimer);
    }
    return stop;
  },

  async getJob(jobId) {
    return toJobState(await request<ServerJobDetail>(`/jobs/${jobId}`));
  },

  async listJobs() {
    return (await request<{ items: JobSummary[] }>(`/jobs?limit=50`)).items;
  },

  renameJob(jobId, title) {
    return request<JobSummary>(`/jobs/${jobId}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ title }),
    });
  },

  async deleteJob(jobId) {
    await request<void>(`/jobs/${jobId}`, { method: "DELETE" });
  },

  cancelJob(jobId) {
    return request<JobSummary>(`/jobs/${jobId}/cancel`, { method: "POST" });
  },

  retryJob(jobId) {
    return request<JobSummary>(`/jobs/${jobId}/retry`, { method: "POST" });
  },

  async health() {
    try {
      return await request<Health>(`/health`);
    } catch {
      return null;
    }
  },

  exportUrl(jobId, fmt) {
    return `${BASE}/jobs/${jobId}/export?fmt=${fmt}`;
  },

  audioUrl(jobId) {
    return `${BASE}/jobs/${jobId}/audio`;
  },
};

// ---------------------------------------------------------------------------
// Mock backend: replays the pipeline with the sample record. File names that
// contain "nospeech" or "fail-lm2" simulate those failures for UI testing.

interface MockJob {
  summary: JobSummary;
  stage: Stage;
  error?: JobError;
  listeners: Set<(e: JobEvent) => void>;
  failAt?: Stage | null;
  /** Bumped on cancel/retry so an older simulated run stops. */
  run?: number;
}
const mockJobs = new Map<string, MockJob>();

const MOCK_STEPS: [Stage, number, string][] = [
  ["validating", 300, "Checking the file"],
  ["normalizing", 400, "Converting audio"],
  ["speech_check", 400, "Detecting speech"],
  ["transcribing", 1600, "Transcribing"],
  ["rechecking", 700, "Double-checking unclear words"],
  ["diarizing", 400, "Labelling speakers"],
  ["raw_saved", 200, "Saving the raw transcript"],
  ["vocabulary", 500, "Learning the meeting's vocabulary"],
  ["refining", 1100, "Refining terminology (window 1 of 1)"],
  ["guarding", 300, "Checking every edit"],
  ["documenting", 1600, "Writing the meeting record"],
  ["verifying", 400, "Verifying the record against the transcript"],
  ["rendering", 200, "Preparing downloads"],
];

const nowIso = () => new Date().toISOString();

function mockSummary(id: string, title: string, file: string, status: JobSummary["status"]): JobSummary {
  return {
    id,
    title,
    source_file: file,
    status,
    stage: status === "completed" ? "completed" : "queued",
    n_decisions: status === "completed" ? SAMPLE_RECORD.decisions.length : 0,
    n_tasks: status === "completed" ? SAMPLE_RECORD.action_items.length : 0,
    duration_s: SAMPLE_RECORD.meta.duration_s,
    created_at: nowIso(),
    updated_at: nowIso(),
  };
}

mockJobs.set("sample", {
  summary: mockSummary("sample", "Weekly sync (sample)", "weekly_sync.mp3", "completed"),
  stage: "completed",
  listeners: new Set(),
});

function mockPartial(job: MockJob): PartialRecord {
  const i = MOCK_STEPS.findIndex(([s]) => s === job.stage);
  const done = (s: Stage) => job.stage === "completed" || (i >= 0 && i > MOCK_STEPS.findIndex(([x]) => x === s));
  const meta = { ...SAMPLE_RECORD.meta, title: job.summary.title, source_file: job.summary.source_file };
  if (job.stage === "completed") return { ...SAMPLE_RECORD, meta };
  const r: PartialRecord = { meta };
  if (done("raw_saved")) r.raw_transcript = SAMPLE_RECORD.raw_transcript;
  if (done("guarding")) {
    r.refined_transcript = SAMPLE_RECORD.refined_transcript;
    r.refinement = SAMPLE_RECORD.refinement;
  }
  return r;
}

function mockEvent(job: MockJob, status: JobEvent["status"], message?: string): JobEvent {
  const i = MOCK_STEPS.findIndex(([s]) => s === job.stage);
  const partial = mockPartial(job);
  return {
    stage: status === "failed" ? "failed" : job.stage,
    status,
    message,
    progress: job.stage === "completed" ? 1 : Math.max(0, i) / MOCK_STEPS.length,
    raw_ready: !!partial.raw_transcript,
    refined_ready: !!partial.refined_transcript,
    record_ready: job.stage === "completed",
    error: job.error,
  };
}

function mockFail(job: MockJob, err: JobError) {
  job.error = err;
  job.summary = {
    ...job.summary,
    status: "failed",
    stage: "failed",
    error_code: err.code as string,
    error_message: err.user_message,
    error_detail: err.detail ?? null,
  };
  job.listeners.forEach((l) => l(mockEvent(job, "failed")));
}

/** Simulate the pipeline from step `from`; a failure at failAt only happens once. */
async function runMock(job: MockJob, from: number) {
  const run = (job.run = (job.run ?? 0) + 1);
  for (const [stage, ms, message] of MOCK_STEPS.slice(from)) {
    if (job.run !== run) return;
    job.stage = stage;
    job.summary.status = "running";
    job.listeners.forEach((l) => l(mockEvent(job, "running", message)));
    await new Promise((r) => setTimeout(r, ms));
    if (job.run !== run) return;
    if (stage === job.failAt) {
      job.failAt = null;
      const code = stage === "speech_check" ? "E_NO_SPEECH" : "E_LM2_FAILED";
      mockFail(job, {
        ...makeError(code),
        stage,
        detail: code === "E_LM2_FAILED" ? "gemma3:12b returned invalid JSON 3 times (mock)." : null,
      });
      return;
    }
  }
  job.stage = "completed";
  job.summary = mockSummary(job.summary.id, job.summary.title, job.summary.source_file, "completed");
  job.listeners.forEach((l) => l(mockEvent(job, "completed")));
}

const mockApi: Api = {
  mock: true,

  async createJob(file, { title }) {
    const id = Math.random().toString(16).slice(2, 10).padEnd(8, "0");
    const d = new Date();
    const summary = mockSummary(
      id,
      title?.trim() || `${file.name} – ${d.getDate()} ${d.toLocaleString("en-GB", { month: "short" })} ${d.getFullYear()}`,
      file.name,
      "queued",
    );
    const failAt: Stage | null = /nospeech/i.test(file.name)
      ? "speech_check"
      : /fail-lm2/i.test(file.name)
        ? "documenting"
        : null;
    const job: MockJob = { summary, stage: "queued", listeners: new Set(), failAt };
    mockJobs.set(id, job);
    runMock(job, 0);
    return { jobId: id, cached: false };
  },

  watch(jobId, onEvent) {
    const job = mockJobs.get(jobId);
    if (!job) return () => {};
    job.listeners.add(onEvent);
    onEvent(mockEvent(job, job.error ? "failed" : job.stage === "completed" ? "completed" : "queued"));
    return () => job.listeners.delete(onEvent);
  },

  async getJob(jobId) {
    const job = mockJobs.get(jobId);
    if (!job) throw new ApiError(makeError("E_NOT_FOUND"));
    return {
      job_id: jobId,
      stage: job.error ? "failed" : job.stage,
      error: job.error ?? null,
      record: mockPartial(job),
    };
  },

  async listJobs() {
    return [...mockJobs.values()].map((j) => j.summary).reverse();
  },

  async renameJob(jobId, title) {
    const job = mockJobs.get(jobId);
    if (!job) throw new ApiError(makeError("E_NOT_FOUND"));
    job.summary = { ...job.summary, title, updated_at: nowIso() };
    return job.summary;
  },

  async deleteJob(jobId) {
    const job = mockJobs.get(jobId);
    if (job && (job.summary.status === "running" || job.summary.status === "queued"))
      throw new ApiError(makeError("E_JOB_RUNNING"));
    mockJobs.delete(jobId);
  },

  async cancelJob(jobId) {
    const job = mockJobs.get(jobId);
    if (!job) throw new ApiError(makeError("E_NOT_FOUND"));
    if (job.summary.status === "running" || job.summary.status === "queued") {
      job.run = (job.run ?? 0) + 1;
      mockFail(job, { ...makeError("E_CANCELLED"), stage: job.stage, detail: "Cancelled by the user." });
    }
    return job.summary;
  },

  async retryJob(jobId) {
    const job = mockJobs.get(jobId);
    if (!job) throw new ApiError(makeError("E_NOT_FOUND"));
    if (job.summary.status !== "failed") return job.summary;
    const from = Math.max(0, MOCK_STEPS.findIndex(([st]) => st === (job.error?.stage ?? job.stage)));
    job.error = undefined;
    job.summary = { ...job.summary, status: "queued", stage: "queued", error_code: null, error_message: null, error_detail: null };
    runMock(job, from);
    return job.summary;
  },

  async health() {
    return null;
  },

  exportUrl() {
    return null;
  },

  audioUrl() {
    return null;
  },
};

const useMock =
  import.meta.env.VITE_MOCK === "true" || new URLSearchParams(location.search).has("mock");

export const api: Api = useMock ? mockApi : httpApi;
