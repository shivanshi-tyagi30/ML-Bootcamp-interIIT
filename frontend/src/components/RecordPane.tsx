import { useEffect, useRef } from "react";
import { isUnspecified } from "../lib/format";
import type { ActionItem, CitedSentence, JobError, PartialRecord, Pointer, Segment, VerifierFlag } from "../lib/types";
import { Empty, Icon, Tabs, Tip, cx } from "./ui";

export type RecordTab = "summary" | "minutes" | "decisions" | "actions" | "proposals";

interface Props {
  record: PartialRecord;
  segments: Map<string, Segment>;
  tab: RecordTab;
  onTab: (t: RecordTab) => void;
  selectedId: string | null;
  onJump: (segmentIds: string[], itemId?: string) => void;
  error?: JobError;
}

/** Evidence chips: one click scrolls the transcript and plays the moment (plan 11.3). */
function Chips({ ids, segments, onJump, itemId }: { ids: string[]; segments: Map<string, Segment>; onJump: Props["onJump"]; itemId?: string }) {
  return (
    <span className="inline-flex flex-wrap gap-1 align-middle">
      {ids.map((id) => {
        const seg = segments.get(id);
        return (
          <Tip
            key={id}
            focusable={false}
            content={seg ? <><span className="text-ink-3">{seg.speaker ?? id}:</span> “{seg.text}”</> : "Segment not found"}
          >
            <button
              onClick={() => onJump([id], itemId)}
              className="rounded-[2px] border border-accent/25 bg-accent-soft px-1.5 py-px font-mono text-[10.5px] font-medium text-accent-deep hover:border-accent"
            >
              {id}
            </button>
          </Tip>
        );
      })}
    </span>
  );
}

function Sentence({ s, segments, onJump }: { s: CitedSentence; segments: Map<string, Segment>; onJump: Props["onJump"] }) {
  return (
    <>
      {s.text} <Chips ids={s.evidence_segment_ids} segments={segments} onJump={onJump} />
    </>
  );
}

const FLAG_TEXT: Record<VerifierFlag, string> = {
  owner_downgraded: "The verifier removed the owner",
  deadline_downgraded: "The verifier removed the deadline",
  pointer_invalid: "The model pointed at words that aren't in the transcript",
  audio_unclear: "The audio check couldn't confirm the words",
};

function Field({
  label,
  value,
  pointer,
  flags,
  kind,
  segments,
}: {
  label: string;
  value: string;
  pointer?: Pointer | null;
  flags: VerifierFlag[];
  kind: "owner" | "deadline";
  segments: Map<string, Segment>;
}) {
  if (isUnspecified(value)) {
    const reason = flags.includes("audio_unclear")
      ? `Audio unclear: the name or time${pointer ? ` (“${pointer.exact_words}”)` : ""} couldn't be confirmed.`
      : flags.includes("pointer_invalid")
        ? "The model's pointer didn't match the transcript, so nothing was copied."
        : "Not stated in the recording.";
    return (
      <div className="flex items-center gap-2">
        <dt className="w-16 text-xs text-ink-3">{label}</dt>
        <dd>
          <Tip content={reason}>
            <span className="inline-flex items-center gap-1 rounded-full border border-dashed border-unspec/70 px-2 py-px text-xs text-unspec">
              Unspecified
              {flags.includes("audio_unclear") && <Icon.alert className="size-3 text-warn" />}
            </span>
          </Tip>
        </dd>
      </div>
    );
  }
  const seg = pointer ? segments.get(pointer.segment_id) : undefined;
  return (
    <div className="flex items-center gap-2">
      <dt className="w-16 text-xs text-ink-3">{label}</dt>
      <dd className="text-sm font-medium">
        <Tip
          content={
            pointer ? (
              <>
                Copied from <span className="font-mono">{pointer.segment_id}</span>
                {seg && <>: “{seg.text}”</>}
              </>
            ) : (
              `${kind === "owner" ? "Owner" : "Deadline"} as stated in the recording.`
            )
          }
          className="cursor-help border-b border-dotted border-ink-3/50"
        >
          {value}
        </Tip>
      </dd>
    </div>
  );
}

function ActionCard({ a, segments, onJump, selected }: { a: ActionItem; segments: Map<string, Segment>; onJump: Props["onJump"]; selected: boolean }) {
  const flags = a.verifier_flags ?? [];
  return (
    <Card id={a.id} selected={selected} onPlay={() => onJump(a.evidence_segment_ids, a.id)}>
      <div className="flex items-start gap-2">
        <span className="mt-0.5 font-mono text-[11px] text-ink-3">{a.id}</span>
        <div className="flex-1">
          <div className="font-medium">{a.task}</div>
          <dl className="mt-2 space-y-1">
            <Field label="Owner" kind="owner" value={a.owner} pointer={a.owner_evidence} flags={flags} segments={segments} />
            <Field label="Due" kind="deadline" value={a.deadline} pointer={a.deadline_evidence} flags={flags} segments={segments} />
          </dl>
          <blockquote className="mt-2.5 border-l-2 border-line pl-2.5 text-sm text-ink-2 italic">“{a.evidence_quote}”</blockquote>
          <div className="mt-2 flex flex-wrap items-center gap-2">
            <Chips ids={a.evidence_segment_ids} segments={segments} onJump={onJump} itemId={a.id} />
            {flags.map((f) => (
              <Tip key={f} content={FLAG_TEXT[f] ?? f}>
                <span className="rounded-[2px] bg-warn-soft px-1.5 py-px text-[10.5px] font-medium text-warn">{f.replace("_", " ")}</span>
              </Tip>
            ))}
          </div>
        </div>
      </div>
    </Card>
  );
}

function Card({ id, selected, onPlay, children }: { id: string; selected: boolean; onPlay: () => void; children: React.ReactNode }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (selected) ref.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [selected]);
  return (
    <div
      ref={ref}
      data-item={id}
      className={cx(
        "group relative rounded-[2px] border bg-surface p-3.5 transition-colors",
        selected ? "border-accent ring-1 ring-accent" : "border-line hover:border-ink-3/40",
      )}
    >
      {children}
      <button
        onClick={onPlay}
        aria-label={`Play ${id} in the recording`}
        title="Play this moment"
        className="absolute top-2.5 right-2.5 grid size-7 place-items-center rounded-full text-ink-3 hover:bg-accent-soft hover:text-accent-deep"
      >
        <Icon.play className="size-3.5" />
      </button>
    </div>
  );
}

export function RecordPane({ record, segments, tab, onTab, selectedId, onJump, error }: Props) {
  const r = record;
  const ready = Array.isArray(r.summary);
  const count = (n?: number) => (n != null ? <span className="ml-1 text-ink-3">{n}</span> : null);

  return (
    <section className="flex min-h-0 flex-col max-md:h-[65vh] max-md:border-t max-md:border-line" aria-label="Meeting record">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-line px-4 py-2.5">
        <h2 className="text-xs font-semibold tracking-wider text-ink-3 uppercase">Record</h2>
        <Tabs
          label="Record section"
          value={tab}
          onChange={onTab}
          items={[
            { value: "summary", label: "Summary" },
            { value: "minutes", label: "Minutes" },
            { value: "decisions", label: <>Decisions{count(r.decisions?.length)}</> },
            { value: "actions", label: <>Action items{count(r.action_items?.length)}</> },
            { value: "proposals", label: <>Not agreed{count(r.open_proposals?.length)}</> },
          ]}
        />
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto p-4">
        {!ready ? (
          <div role="alert" className="rounded-[2px] border border-bad/30 bg-bad-soft p-4 text-sm">
            <div className="font-medium text-bad">{error?.user_message ?? "The meeting record isn't available."}</div>
            <p className="mt-1 text-ink-2">The raw and refined transcripts are still available on the left and in Downloads.</p>
            {error?.code && <div className="mt-1 font-mono text-[11px] text-ink-3">{error.code}</div>}
          </div>
        ) : tab === "summary" ? (
          r.summary!.length ? (
            <div className="space-y-2.5 text-[15px] leading-relaxed">
              {r.summary!.map((s, i) => (
                <p key={i}>
                  <Sentence s={s} segments={segments} onJump={onJump} />
                </p>
              ))}
            </div>
          ) : (
            <Empty>Nothing to summarize.</Empty>
          )
        ) : tab === "minutes" ? (
          r.minutes?.length ? (
            <div className="space-y-5">
              {r.minutes.map((t, i) => (
                <div key={i}>
                  <h3 className="mb-1.5 text-sm font-semibold">{t.topic}</h3>
                  <ul className="list-disc space-y-1.5 pl-5 text-[15px] leading-relaxed marker:text-ink-3">
                    {t.points.map((p, j) => (
                      <li key={j}>
                        <Sentence s={p} segments={segments} onJump={onJump} />
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          ) : (
            <Empty>No minutes were generated.</Empty>
          )
        ) : tab === "decisions" ? (
          r.decisions?.length ? (
            <div className="space-y-2.5">
              {r.decisions.map((d) => (
                <Card key={d.id} id={d.id} selected={selectedId === d.id} onPlay={() => onJump(d.evidence_segment_ids, d.id)}>
                  <div className="flex items-start gap-2 pr-8">
                    <span className="mt-0.5 font-mono text-[11px] text-ink-3">{d.id}</span>
                    <div>
                      <div className="font-medium">{d.decision}</div>
                      <div className="mt-1.5 text-sm text-ink-2">
                        <span className="text-xs text-ink-3">Agreement: </span>“{d.agreement_evidence}”
                      </div>
                      <div className="mt-2">
                        <Chips ids={d.evidence_segment_ids} segments={segments} onJump={onJump} itemId={d.id} />
                      </div>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          ) : (
            <Empty>No decisions were reached.</Empty>
          )
        ) : tab === "actions" ? (
          r.action_items?.length ? (
            <div className="space-y-2.5">
              {r.action_items.map((a) => (
                <ActionCard key={a.id} a={a} segments={segments} onJump={onJump} selected={selectedId === a.id} />
              ))}
            </div>
          ) : (
            <Empty>No action items were identified.</Empty>
          )
        ) : (
          <>
            <p className="mb-3 flex items-start gap-1.5 text-xs text-ink-3">
              <Icon.info className="mt-px size-3.5" />
              Ideas raised without explicit agreement. These are not decisions.
            </p>
            {r.open_proposals?.length ? (
              <ul className="space-y-2">
                {r.open_proposals.map((p, i) => (
                  <li key={i} className="rounded-[2px] border border-dashed border-line bg-surface p-3 text-sm">
                    <span className="mr-2 rounded-[2px] bg-raised px-1.5 py-px text-[10.5px] font-medium tracking-wide text-ink-2 uppercase">Proposal</span>
                    {p.proposal} <Chips ids={p.evidence_segment_ids} segments={segments} onJump={onJump} />
                  </li>
                ))}
              </ul>
            ) : (
              <Empty>No open proposals.</Empty>
            )}
          </>
        )}
      </div>
    </section>
  );
}
