import { useEffect, useRef, useState } from "react";
import { ApiError, type Api } from "../lib/api";
import { makeError } from "../lib/errors";
import { fmtTime } from "../lib/format";
import { type JobError, type PartialRecord, type Segment, type Stage } from "../lib/types";
import { Brand, Button, Icon, cx } from "./ui";

// Plan 11.2: the user-facing steps, each covering one or more backend stages.
const STEPS: { label: string; stages: Stage[]; doing: string }[] = [
  { label: "Checking file", stages: ["uploaded", "queued", "validating", "normalizing", "speech_check"], doing: "Making sure the file is audio and that someone is speaking." },
  { label: "Transcribing", stages: ["transcribing"], doing: "Writing down every word, with timestamps." },
  { label: "Double-checking", stages: ["rechecking", "diarizing", "raw_saved"], doing: "Listening again to names, numbers and anything unclear." },
  { label: "Refining", stages: ["vocabulary", "refining", "guarding"], doing: "Fixing misheard jargon. Numbers, names and “not” stay untouched." },
  { label: "Writing record", stages: ["documenting"], doing: "Drafting the summary, minutes, decisions and action items." },
  { label: "Verifying", stages: ["verifying", "rendering"], doing: "Checking every claim against the transcript before you see it." },
];

interface Props {
  api: Api;
  jobId: string;
  fileName: string;
  /** Finished, or failed late with transcripts available (error set). */
  onDone: (record: PartialRecord, error?: JobError) => void;
  onFail: (error: JobError) => void;
  /** Go back to the upload screen; the job keeps running on the server. */
  onLeave: () => void;
}

export function ProcessingScreen({ api, jobId, fileName, onDone, onFail, onLeave }: Props) {
  const [stage, setStage] = useState<Stage>("uploaded");
  const [failedStage, setFailedStage] = useState<Stage | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [progress, setProgress] = useState(0);
  const [message, setMessage] = useState<string | null>(null);
  const [preview, setPreview] = useState<Segment[] | null>(null);
  const [cancelling, setCancelling] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const cb = useRef({ onDone, onFail });
  cb.current = { onDone, onFail };

  useEffect(() => {
    let fetchedPreview = false;
    const finish = async (error?: JobError) => {
      try {
        const s = await api.getJob(jobId);
        const err = error ?? s.error ?? undefined;
        if (err && !s.record.raw_transcript?.length) cb.current.onFail(err);
        else cb.current.onDone(s.record, err);
      } catch (e) {
        cb.current.onFail(error ?? (e instanceof ApiError ? e.jobError : makeError("E_INTERNAL")));
      }
    };

    let finished = false;
    let unsubscribe = () => {};
    const stop = () => {
      finished = true;
      unsubscribe();
    };
    unsubscribe = api.watch(jobId, (e) => {
      if (finished) return;
      if (e.warnings?.length) setWarnings(e.warnings);
      if (e.progress != null) setProgress(e.progress);
      if (e.message) setMessage(e.message);
      if (e.status === "failed") {
        setFailedStage(e.error?.stage ?? null);
        stop();
        finish(e.error ?? makeError("E_INTERNAL"));
        return;
      }
      setStage(e.stage);
      if (e.status === "completed") {
        stop();
        finish();
        return;
      }
      // Show the raw transcript as soon as it exists (plan 11.2).
      if (!fetchedPreview && e.raw_ready) {
        fetchedPreview = true;
        api.getJob(jobId).then((s) => s.record.raw_transcript && setPreview(s.record.raw_transcript), () => {});
      }
    });
    if (finished) unsubscribe();
    return stop;
  }, [api, jobId]);

  const cancel = async () => {
    setCancelling(true);
    setActionError(null);
    try {
      await api.cancelJob(jobId); // the progress stream then reports the job as cancelled
    } catch (e) {
      setCancelling(false);
      setActionError(e instanceof ApiError ? e.jobError.user_message : "Couldn't cancel.");
    }
  };

  const current = failedStage ?? stage;
  const activeStep = STEPS.findIndex((s) => s.stages.includes(current));

  return (
    <div className="flex min-h-full flex-col bg-bg">
      <header className="flex items-center justify-between border-b border-ink bg-surface px-5 py-3 sm:px-10">
        <Brand compact />
        <div className="flex items-center gap-2">
          <span className="font-mono text-[10.5px] tracking-[0.14em] text-ink-3 max-sm:hidden">JOB {jobId.toUpperCase()}</span>
          <Button variant="ghost" size="sm" onClick={onLeave} title="Processing continues; reopen it from Recent meetings">
            Back to home
          </Button>
          {!failedStage && (
            <Button size="sm" onClick={cancel} disabled={cancelling}>
              <Icon.x className="size-3.5" /> {cancelling ? "Cancelling…" : "Cancel"}
            </Button>
          )}
        </div>
      </header>
    <div className="mx-auto flex w-full max-w-4xl flex-1 flex-col px-5 py-12 sm:px-10">
      <div className="mb-3 flex items-center gap-2.5 font-mono text-[11px] tracking-[0.18em] text-ink-2">
        <span className="checker" /> {failedStage ? "STOPPED" : "IN PROGRESS"}
      </div>
      <h1 className="text-[clamp(34px,4.4vw,56px)] leading-[1.02] font-bold tracking-[-0.04em]">
        {failedStage ? "Something went wrong with" : "Listening to"}
        <span className="block truncate font-serif font-normal tracking-[-0.01em] italic">{fileName}</span>
      </h1>
      <p className="mt-4 min-h-[1.5em] text-[16px] text-ink-2">
        {failedStage
          ? "We kept everything that finished before the problem."
          : stage === "queued"
            ? "Another recording is being processed. Yours starts as soon as it finishes."
            : STEPS[Math.max(0, activeStep)]?.doing}
      </p>
      <div className="mt-6 flex items-center gap-3 font-mono text-[11px] tracking-[0.08em] text-ink-3" aria-live="polite">
        <span className="tabular-nums text-ink">{Math.round(progress * 100)}%</span>
        <span className="h-px flex-1 bg-line">
          <span className="block h-px bg-ink transition-[width] duration-500" style={{ width: `${progress * 100}%` }} />
        </span>
        <span className="truncate">{(message ?? "Starting").toUpperCase()}</span>
      </div>

      <ol className="mt-6 grid grid-cols-2 gap-px border border-ink bg-ink sm:grid-cols-6" aria-label="Progress">
        {STEPS.map((s, i) => {
          const state = i < activeStep ? "done" : i === activeStep ? (failedStage ? "failed" : "active") : "todo";
          return (
            <li
              key={s.label}
              aria-current={state === "active" ? "step" : undefined}
              className={cx(
                "relative flex flex-col gap-3 px-3 py-3",
                state === "done" && "bg-accent text-accent-ink",
                state === "active" && "bg-surface",
                state === "failed" && "bg-bad-soft text-bad",
                state === "todo" && "bg-surface text-ink-3",
              )}
            >
              {state === "active" && <span className="absolute inset-x-0 top-0 h-[3px] animate-pulse bg-accent" />}
              <span className="flex items-center justify-between font-mono text-[10.5px] tracking-[0.1em]">
                {String(i + 1).padStart(2, "0")}
                {state === "done" && <Icon.check className="size-3.5" />}
                {state === "failed" && <Icon.x className="size-3.5" />}
              </span>
              <span className={cx("text-[13px] leading-tight", state !== "todo" && "font-bold")}>{s.label}</span>
            </li>
          );
        })}
      </ol>

      {actionError && (
        <p role="alert" className="mt-6 rounded-[2px] bg-bad-soft px-3 py-2 text-xs text-bad">{actionError}</p>
      )}

      {warnings.includes("W_NON_ENGLISH") && (
        <p className="mt-6 rounded-[2px] bg-warn-soft px-3 py-2 text-xs text-warn">
          This recording may not be in English. Processing continues, but accuracy may be lower.
        </p>
      )}

      <section className="mt-12 flex min-h-0 flex-1 flex-col">
        <h2 className="mb-2 font-mono text-[11px] tracking-[0.14em] text-ink-3">RAW TRANSCRIPT · FIRST DRAFT</h2>
        {preview ? (
          <div className="max-h-[46vh] overflow-y-auto border border-ink/15 bg-surface p-5 text-[15px] leading-relaxed">
            {preview.map((s) => (
              <p key={s.id} className="mb-2">
                <span className="mr-2 font-mono text-[11px] text-ink-3">{fmtTime(s.start)}</span>
                {s.speaker && <span className="mr-1.5 text-xs font-medium text-ink-2">{s.speaker}</span>}
                {s.text}
              </p>
            ))}
          </div>
        ) : (
          <div className="border border-dashed border-ink/20 px-4 py-10 text-center text-sm text-ink-3">
            The first draft of the transcript shows up here as soon as it's written.
          </div>
        )}
      </section>
    </div>
    </div>
  );
}
