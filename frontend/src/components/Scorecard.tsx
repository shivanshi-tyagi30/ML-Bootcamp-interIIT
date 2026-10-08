import type { Fidelity } from "../lib/types";
import { Tip, cx } from "./ui";

function Stat({ label, value, tip, tone }: { label: string; value: string; tip: string; tone?: "ok" | "bad" | "warn" }) {
  return (
    <Tip content={tip} className="flex items-baseline gap-1.5 rounded-[2px] px-2 py-1 hover:bg-raised">
      <span className="text-xs text-ink-3">{label}</span>
      <span
        className={cx(
          "text-sm font-semibold tabular-nums",
          tone === "ok" && "text-ok",
          tone === "bad" && "text-bad",
          tone === "warn" && "text-warn",
        )}
      >
        {value}
      </span>
    </Tip>
  );
}

/** Plan 11.3: accuracy visible at a glance. */
export function Scorecard({ f }: { f: Fidelity }) {
  const items = [];
  if (f.edits_accepted != null)
    items.push(
      <Stat key="e" label="Edits" value={`${f.edits_accepted} applied · ${f.edits_rejected ?? 0} blocked`}
        tip="Terminology fixes proposed by LM1. Blocked edits failed a safety check and were not applied (see the Diff view)." />,
    );
  if (f.disputed_words != null)
    items.push(
      <Stat key="d" label="Disputed words" value={String(f.disputed_words)} tone={f.disputed_words ? "warn" : undefined}
        tip="Words where the second speech model heard something different. Facts resting on them become “Unspecified”." />,
    );
  if (f.transcript_coverage_pct != null)
    items.push(
      <Stat key="c" label="Coverage" value={`${Math.round(f.transcript_coverage_pct)}%`}
        tip="Share of transcript lines cited by the summary or minutes. Uncited lines are dimmed." />,
    );
  if (!items.length) return null;
  return <div className="flex flex-wrap items-center gap-x-1 gap-y-0.5">{items}</div>;
}

export function Legend() {
  const sw = "inline-block h-3 w-5 rounded-sm align-middle";
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-ink-3">
      <span><span className={cx(sw, "bg-ok-soft ring-1 ring-ok/40")} /> applied edit</span>
      <span><span className={cx(sw, "bg-bad-soft ring-1 ring-bad/40")} /> blocked edit</span>
      <span><span className={cx(sw, "border-b-2 border-dotted border-warn")} /> disputed word</span>
      <span><span className={cx(sw, "border border-dashed border-unspec/60")} /> unspecified</span>
    </div>
  );
}
