import { ERROR_MESSAGES, makeError } from "./errors";
import { SAMPLE_RECORD } from "./sampleRecord";
import type { ExportFormat, JobError, JobEvent, JobState, PartialRecord, Stage } from "./types";

export interface Api {
  mock: boolean;
  createJob(file: File, glossary: string): Promise<string>;
  /** Calls onEvent for every stage change until completed/failed. Returns an unsubscribe fn. */
  watch(jobId: string, onEvent: (e: JobEvent) => void): () => void;
  getJob(jobId: string): Promise<JobState>;
  /** Backend download URL, or null when the client should build the file itself. */
  exportUrl(jobId: string, fmt: ExportFormat): string | null;
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
    const err = body.error ?? body.detail ?? body;
    if (err?.code) return { code: err.code, stage: err.stage, user_message: err.user_message ?? ERROR_MESSAGES.E_INTERNAL };
  } catch {
    /* not JSON */
  }
  return makeError("E_INTERNAL", `Server error (${res.status}).`);
}

const httpApi: Api = {
  mock: false,

  async createJob(file, glossary) {
    const form = new FormData();
    form.append("file", file);
    if (glossary.trim()) form.append("glossary", glossary.trim());
    let res: Response;
    try {
      res = await fetch(`${BASE}/jobs`, { method: "POST", body: form });
    } catch {
      throw new ApiError(makeError("E_NETWORK"));
    }
    if (!res.ok) throw new ApiError(await readError(res));
    const body = await res.json();
    return body.job_id as string;
  },

  watch(jobId, onEvent) {
    let closed = false;
    let pollTimer: number | undefined;
    let lastStage: Stage | undefined;

    const emit = (e: JobEvent) => {
      if (closed) return;
      lastStage = e.stage;
      onEvent(e);
      if (e.stage === "completed" || e.stage === "failed") stop();
    };

    // Fallback if SSE drops (proxies, restarts): poll the job state.
    const poll = async () => {
      if (closed) return;
      try {
        const s = await httpApi.getJob(jobId);
        if (s.stage !== lastStage) emit({ stage: s.stage, error: s.error ?? undefined, warnings: s.warnings });
      } catch {
        /* keep trying */
      }
      if (!closed) pollTimer = window.setTimeout(poll, 2000);
    };

    const es = new EventSource(`${BASE}/jobs/${jobId}/events`);
    es.onmessage = (m) => {
      try {
        emit(JSON.parse(m.data) as JobEvent);
      } catch {
        /* ignore malformed event */
      }
    };
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
    let res: Response;
    try {
      res = await fetch(`${BASE}/jobs/${jobId}`);
    } catch {
      throw new ApiError(makeError("E_NETWORK"));
    }
    if (!res.ok) throw new ApiError(await readError(res));
    return res.json();
  },

  exportUrl(jobId, fmt) {
    return `${BASE}/jobs/${jobId}/export?fmt=${fmt}`;
  },
};

// ---------------------------------------------------------------------------
// Mock backend: replays the pipeline with the sample record. File names that
// contain "nospeech" or "fail-lm2" simulate those failures for UI testing.

interface MockJob {
  file: string;
  stage: Stage;
  error?: JobError;
  listeners: Set<(e: JobEvent) => void>;
}
const mockJobs = new Map<string, MockJob>();

const MOCK_STEPS: [Stage, number][] = [
  ["validating", 400],
  ["normalizing", 500],
  ["speech_check", 500],
  ["transcribing", 1800],
  ["rechecking", 900],
  ["refining", 1300],
  ["guarding", 400],
  ["documenting", 1800],
  ["verifying", 600],
  ["rendering", 300],
];

function mockPartial(job: MockJob): PartialRecord {
  const i = MOCK_STEPS.findIndex(([s]) => s === job.stage);
  const done = (s: Stage) => job.stage === "completed" || (i >= 0 && i > MOCK_STEPS.findIndex(([x]) => x === s));
  const meta = { ...SAMPLE_RECORD.meta, source_file: job.file };
  const r: PartialRecord = { meta };
  if (done("rechecking")) r.raw_transcript = SAMPLE_RECORD.raw_transcript;
  if (done("guarding")) {
    r.refined_transcript = SAMPLE_RECORD.refined_transcript;
    r.refinement = SAMPLE_RECORD.refinement;
  }
  if (job.stage === "completed") return { ...SAMPLE_RECORD, meta };
  return r;
}

const mockApi: Api = {
  mock: true,

  async createJob(file) {
    const id = `mock-${Math.random().toString(36).slice(2, 8)}`;
    const job: MockJob = { file: file.name, stage: "uploaded", listeners: new Set() };
    mockJobs.set(id, job);
    const failAt: Stage | null = /nospeech/i.test(file.name)
      ? "speech_check"
      : /fail-lm2/i.test(file.name)
        ? "documenting"
        : null;

    (async () => {
      for (const [stage, ms] of MOCK_STEPS) {
        job.stage = stage;
        job.listeners.forEach((l) => l({ stage }));
        await new Promise((r) => setTimeout(r, ms));
        if (stage === failAt) {
          job.error = { ...makeError(stage === "speech_check" ? "E_NO_SPEECH" : "E_LM2_FAILED"), stage };
          job.listeners.forEach((l) => l({ stage: "failed", error: job.error }));
          // keep job.stage at the failing stage so partial results reflect it
          return;
        }
      }
      job.stage = "completed";
      job.listeners.forEach((l) => l({ stage: "completed" }));
    })();
    return id;
  },

  watch(jobId, onEvent) {
    const job = mockJobs.get(jobId);
    if (!job) return () => {};
    job.listeners.add(onEvent);
    onEvent({ stage: job.stage });
    return () => job.listeners.delete(onEvent);
  },

  async getJob(jobId) {
    const job = mockJobs.get(jobId);
    if (jobId === "sample") return { job_id: "sample", stage: "completed", record: SAMPLE_RECORD };
    if (!job) throw new ApiError(makeError("E_INTERNAL", "Job not found."));
    return {
      job_id: jobId,
      stage: job.error ? "failed" : job.stage,
      error: job.error ?? null,
      record: mockPartial(job),
    };
  },

  exportUrl() {
    return null;
  },
};

const useMock =
  import.meta.env.VITE_MOCK === "true" || new URLSearchParams(location.search).has("mock");

export const api: Api = useMock ? mockApi : httpApi;
