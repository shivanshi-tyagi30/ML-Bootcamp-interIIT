import { useEffect, useRef, useState } from "react";
import { ApiError, type Api } from "../lib/api";
import { makeError } from "../lib/errors";
import { fmtTime } from "../lib/format";
import { PIPELINE_ORDER, type JobError, type PartialRecord, type Segment, type Stage } from "../lib/types";
import { Icon, cx } from "./ui";

// Plan 11.2: the user-facing steps, each covering one or more backend stages.
const STEPS: { label: string; stages: Stage[] }[] = [
  { label: "Checking file", stages: ["uploaded", "validating", "normalizing", "speech_check"] },
  { label: "Transcribing", stages: ["transcribing"] },
  { label: "Double-checking", stages: ["rechecking"] },
  { label: "Refining", stages: ["refining", "guarding"] },
  { label: "Writing record", stages: ["documenting"] },
  { label: "Verifying", stages: ["verifying", "rendering"] },
];

const idx = (s: Stage) => PIPELINE_ORDER.indexOf(s);

interface Props {
  api: Api;
  jobId: string;
  fileName: string;
  /** Finished, or failed late with transcripts available (error set). */
  onDone: (record: PartialRecord, error?: JobError) => void;
  onFail: (error: JobError) => void;
}

export function ProcessingScreen({ api, jobId, fileName, onDone, onFail }: Props) {
  const [stage, setStage] = useState<Stage>("uploaded");
  const [failedStage, setFailedStage] = useState<Stage | null>(null);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [preview, setPreview] = useState<Segment[] | null>(null);
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
      if (e.stage === "failed") {
        setFailedStage(e.error?.stage ?? null);
        stop();
        finish(e.error ?? makeError("E_INTERNAL"));
        return;
      }
      setStage(e.stage);
      if (e.stage === "completed") {
        stop();
        finish();
        return;
      }
      // Show the raw transcript as soon as it exists (plan 11.2).
      if (!fetchedPreview && idx(e.stage) > idx("rechecking")) {
        fetchedPreview = true;
        api.getJob(jobId).then((s) => s.record.raw_transcript && setPreview(s.record.raw_transcript), () => {});
      }
    });
    if (finished) unsubscribe();
    return stop;
  }, [api, jobId]);

  const current = failedStage ?? stage;
  const activeStep = STEPS.findIndex((s) => s.stages.includes(current));

  return (
    <div className="mx-auto flex min-h-full max-w-3xl flex-col px-4 py-12">
      <div className="mb-1 text-sm text-ink-3">Processing</div>
      <h1 className="mb-8 truncate text-xl font-semibold">{fileName}</h1>

      <ol className="grid grid-cols-6 gap-2" aria-label="Progress">
        {STEPS.map((s, i) => {
          const state = i < activeStep ? "done" : i === activeStep ? (failedStage ? "failed" : "active") : "todo";
          return (
            <li key={s.label} className="flex flex-col gap-2" aria-current={state === "active" ? "step" : undefined}>
              <div
                className={cx(
                  "h-1.5 rounded-full",
                  state === "done" && "bg-accent",
                  state === "active" && "animate-pulse bg-accent/60",
                  state === "failed" && "bg-bad",
                  state === "todo" && "bg-line",
                )}
              />
              <div
                className={cx(
                  "flex items-center gap-1 text-xs",
                  state === "todo" ? "text-ink-3" : state === "failed" ? "font-medium text-bad" : "font-medium text-ink",
                )}
              >
                {state === "done" && <Icon.check className="size-3.5 text-accent" />}
                {state === "failed" && <Icon.x className="size-3.5" />}
                {s.label}
              </div>
            </li>
          );
        })}
      </ol>

      {warnings.includes("W_NON_ENGLISH") && (
        <p className="mt-6 rounded-md bg-warn-soft px-3 py-2 text-xs text-warn">
          This recording may not be in English. Processing continues, but accuracy may be lower.
        </p>
      )}

      <section className="mt-10 flex min-h-0 flex-1 flex-col">
        <h2 className="mb-2 text-sm font-medium text-ink-2">Raw transcript</h2>
        {preview ? (
          <div className="max-h-[50vh] overflow-y-auto rounded-lg border border-line bg-surface p-4 text-sm leading-relaxed">
            {preview.map((s) => (
              <p key={s.id} className="mb-2">
                <span className="mr-2 font-mono text-[11px] text-ink-3">{fmtTime(s.start)}</span>
                {s.speaker && <span className="mr-1.5 text-xs font-medium text-ink-2">{s.speaker}</span>}
                {s.text}
              </p>
            ))}
          </div>
        ) : (
          <div className="rounded-lg border border-dashed border-line px-4 py-10 text-center text-sm text-ink-3">
            The transcript will appear here once speech recognition finishes.
          </div>
        )}
      </section>
    </div>
  );
}
