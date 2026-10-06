import { useCallback, useEffect, useRef, useState } from "react";
import { ApiError, type Api } from "../lib/api";
import { fmtTime } from "../lib/format";
import type { JobSummary } from "../lib/types";
import { Icon, cx } from "./ui";

const STATUS: Record<JobSummary["status"], { label: string; cls: string }> = {
  completed: { label: "Ready", cls: "bg-ink text-surface" },
  running: { label: "Processing", cls: "bg-accent text-accent-ink" },
  queued: { label: "Queued", cls: "border border-ink/30 text-ink-2" },
  failed: { label: "Failed", cls: "bg-bad-soft text-bad" },
};

function when(iso: string): string {
  const d = new Date(iso);
  return isNaN(+d) ? "" : d.toLocaleDateString("en-GB", { day: "numeric", month: "short" }) + " " +
    d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
}

function Row({
  job,
  onOpen,
  onRename,
  onDelete,
}: {
  job: JobSummary;
  onOpen: () => void;
  onRename: (title: string) => Promise<void>;
  onDelete: () => Promise<void>;
}) {
  const [editing, setEditing] = useState(false);
  const [confirm, setConfirm] = useState(false);
  const [draft, setDraft] = useState(job.title);
  const input = useRef<HTMLInputElement>(null);
  const st = STATUS[job.status] ?? STATUS.queued;
  const busy = job.status === "running" || job.status === "queued";

  useEffect(() => {
    if (editing) input.current?.select();
  }, [editing]);

  const save = async () => {
    const t = draft.trim();
    setEditing(false);
    if (t && t !== job.title) await onRename(t);
    else setDraft(job.title);
  };

  return (
    <li className="group flex items-center gap-3 border-b border-line px-3 py-2.5 last:border-b-0">
      <span className={cx("shrink-0 px-1.5 py-px font-mono text-[9.5px] tracking-[0.08em] uppercase", st.cls)}>
        {st.label}
      </span>
      <div className="min-w-0 flex-1">
        {editing ? (
          <input
            ref={input}
            value={draft}
            maxLength={100}
            aria-label="Meeting name"
            onChange={(e) => setDraft(e.target.value)}
            onBlur={save}
            onKeyDown={(e) => {
              if (e.key === "Enter") save();
              if (e.key === "Escape") {
                setDraft(job.title);
                setEditing(false);
              }
            }}
            className="w-full border-b border-ink bg-transparent text-[14px] font-medium focus:outline-none"
          />
        ) : (
          <button onClick={onOpen} className="block w-full truncate text-left text-[14px] font-medium hover:underline">
            {job.title}
          </button>
        )}
        <div className="truncate font-mono text-[10.5px] text-ink-3">
          {when(job.created_at)}
          {job.duration_s ? ` · ${fmtTime(job.duration_s)}` : ""}
          {job.status === "completed" &&
            ` · ${job.n_decisions} decision${job.n_decisions === 1 ? "" : "s"} · ${job.n_tasks} task${job.n_tasks === 1 ? "" : "s"}`}
          {job.status === "failed" && job.error_code ? ` · ${job.error_code}` : ""}
        </div>
      </div>
      <div className="flex shrink-0 items-center gap-1 opacity-60 group-focus-within:opacity-100 group-hover:opacity-100">
        <button
          onClick={() => setEditing(true)}
          aria-label={`Rename ${job.title}`}
          title="Rename"
          className="grid size-7 place-items-center text-ink-2 hover:bg-raised hover:text-ink"
        >
          <svg viewBox="0 0 24 24" className="size-3.5" fill="none" stroke="currentColor" strokeWidth={2} aria-hidden>
            <path d="M4 20h4L19 9l-4-4L4 16v4zM14 6l4 4" />
          </svg>
        </button>
        {confirm ? (
          <button
            onClick={async () => {
              setConfirm(false);
              await onDelete();
            }}
            onBlur={() => setConfirm(false)}
            autoFocus
            className="h-7 bg-bad px-2 text-[11px] font-bold text-white"
          >
            Delete?
          </button>
        ) : (
          <button
            onClick={() => setConfirm(true)}
            disabled={busy}
            aria-label={`Delete ${job.title}`}
            title={busy ? "Can't delete while processing" : "Delete"}
            className="grid size-7 place-items-center text-ink-2 hover:bg-bad-soft hover:text-bad disabled:opacity-30"
          >
            <Icon.x className="size-3.5" />
          </button>
        )}
      </div>
    </li>
  );
}

/** Past meetings from GET /api/jobs, with open, rename and delete. */
export function RecentMeetings({ api, onOpen }: { api: Api; onOpen: (jobId: string) => void }) {
  const [jobs, setJobs] = useState<JobSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(() => {
    api.listJobs().then(
      (j) => {
        setJobs(j);
        setError(null);
      },
      (e) => setError(e instanceof ApiError ? e.jobError.user_message : "Couldn't load past meetings."),
    );
  }, [api]);

  useEffect(() => {
    load();
    // Refresh while something is processing so its status updates.
    const t = window.setInterval(load, 5000);
    return () => window.clearInterval(t);
  }, [load]);

  if (error) return <p className="mt-8 font-mono text-[11px] text-ink-3">{error}</p>;
  if (!jobs?.length) return null;

  return (
    <section className="mt-10 max-w-[560px]" aria-label="Recent meetings">
      <h2 className="mb-2 flex items-center gap-2.5 font-mono text-[11px] tracking-[0.18em] text-ink-2">
        <span className="checker" /> RECENT MEETINGS
      </h2>
      <ul className="max-h-[260px] overflow-y-auto border border-ink/15 bg-surface">
        {jobs.map((j) => (
          <Row
            key={j.id}
            job={j}
            onOpen={() => onOpen(j.id)}
            onRename={async (title) => {
              try {
                const updated = await api.renameJob(j.id, title);
                setJobs((js) => js?.map((x) => (x.id === j.id ? updated : x)) ?? null);
              } catch (e) {
                setError(e instanceof ApiError ? e.jobError.user_message : "Rename failed.");
              }
            }}
            onDelete={async () => {
              try {
                await api.deleteJob(j.id);
                setJobs((js) => js?.filter((x) => x.id !== j.id) ?? null);
              } catch (e) {
                setError(e instanceof ApiError ? e.jobError.user_message : "Delete failed.");
              }
            }}
          />
        ))}
      </ul>
    </section>
  );
}
