import { useVirtualizer } from "@tanstack/react-virtual";
import { memo, useEffect, useMemo, useRef } from "react";
import { REJECT_REASONS, annotate, type Piece, type TranscriptMode } from "../lib/annotate";
import { fmtTime } from "../lib/format";
import type { Edit, RejectedEdit, Segment } from "../lib/types";
import { Tabs, Tip, cx } from "./ui";

interface Props {
  raw: Segment[];
  accepted: Edit[];
  rejected: RejectedEdit[];
  hasRefinement: boolean;
  mode: TranscriptMode;
  onMode: (m: TranscriptMode) => void;
  activeId: string | null;
  scrollTo: { id: string; nonce: number } | null;
  onSeek: (t: number) => void;
}

const pct = (c: number) => `${Math.round(c * 100)}%`;

function PieceView({ p, mode }: { p: Piece; mode: TranscriptMode }) {
  switch (p.kind) {
    case "text":
      return <>{p.text}</>;
    case "disputed":
      return (
        <Tip
          className="cursor-help border-b-2 border-dotted border-warn"
          content={
            <>
              <div className="font-medium text-warn">{p.word.alt != null ? "Disputed word" : "Unclear word"}</div>
              {p.word.alt != null ? (
                <>The second speech model heard “{p.word.alt || "nothing"}”.</>
              ) : (
                <>The speech model wasn't sure about this word.</>
              )}
              {p.word.conf != null && <> Confidence {pct(p.word.conf)}.</>}
            </>
          }
        >
          {p.text}
        </Tip>
      );
    case "accepted": {
      const e = p.edit;
      const tip = (
        <>
          <div className="font-medium text-ok">Applied edit · {e.category.replace("_", " ")}</div>
          “{e.original}” → “{e.replacement}”
          {e.rationale && <div className="text-ink-2">{e.rationale}</div>}
          <div className="text-ink-3">Confidence {pct(e.confidence)}</div>
        </>
      );
      if (mode === "diff")
        return (
          <Tip content={tip} className="cursor-help">
            <del className="rounded-sm bg-bad-soft/60 px-0.5 text-ink-3 decoration-ink-3">{e.original}</del>
            <ins className="ml-0.5 rounded-sm bg-ok-soft px-0.5 text-ok no-underline">{e.replacement}</ins>
          </Tip>
        );
      return (
        <Tip content={tip} className="cursor-help rounded-sm bg-ok-soft/70 px-0.5">
          {e.replacement}
        </Tip>
      );
    }
    case "rejected": {
      const e = p.edit;
      return (
        <Tip
          className="cursor-help rounded-sm bg-bad-soft px-0.5 decoration-bad decoration-wavy underline-offset-4 underline"
          content={
            <>
              <div className="font-medium text-bad">Blocked edit</div>
              LM1 proposed “{e.replacement}”. Blocked because{" "}
              {REJECT_REASONS[e.reject_reason ?? ""] ?? e.reject_reason ?? "it failed a safety check"}.
            </>
          }
        >
          {e.original}
        </Tip>
      );
    }
  }
}

const Row = memo(function Row({
  seg,
  pieces,
  mode,
  active,
  onSeek,
}: {
  seg: Segment;
  pieces: Piece[];
  mode: TranscriptMode;
  active: boolean;
  onSeek: (t: number) => void;
}) {
  return (
    <div
      className={cx(
        "grid grid-cols-[3.5rem_1fr] gap-x-3 border-l-2 px-4 py-2.5",
        active ? "border-accent bg-accent-soft/50" : "border-transparent",
      )}
    >
      <button
        onClick={() => onSeek(seg.start)}
        className="h-fit pt-0.5 text-left font-mono text-[11px] text-ink-3 hover:text-accent-deep"
        title={`Play from ${fmtTime(seg.start)}`}
      >
        {fmtTime(seg.start)}
        <div className="text-[10px] opacity-70">{seg.id}</div>
      </button>
      <div>
        {seg.speaker && <div className="mb-0.5 text-xs font-medium text-ink-2">{seg.speaker}</div>}
        <p className="text-[15px] leading-relaxed">
          {pieces.map((p, i) => (
            <PieceView key={i} p={p} mode={mode} />
          ))}
        </p>
      </div>
    </div>
  );
});

export function TranscriptPane({ raw, accepted, rejected, hasRefinement, mode, onMode, activeId, scrollTo, onSeek }: Props) {
  const parent = useRef<HTMLDivElement>(null);
  const effectiveMode: TranscriptMode = hasRefinement ? mode : "raw";

  const bySeg = useMemo(() => {
    const group = <T extends Edit>(xs: T[]) => {
      const m = new Map<string, T[]>();
      for (const x of xs) m.set(x.segment_id, [...(m.get(x.segment_id) ?? []), x]);
      return m;
    };
    return { acc: group(accepted), rej: group(rejected) };
  }, [accepted, rejected]);

  const pieces = useMemo(
    () => raw.map((s) => annotate(s, bySeg.acc.get(s.id) ?? [], bySeg.rej.get(s.id) ?? [], effectiveMode)),
    [raw, bySeg, effectiveMode],
  );
  const index = useMemo(() => new Map(raw.map((s, i) => [s.id, i])), [raw]);

  const v = useVirtualizer({
    count: raw.length,
    getScrollElement: () => parent.current,
    estimateSize: () => 72,
    overscan: 8,
  });

  useEffect(() => {
    if (!scrollTo) return;
    const i = index.get(scrollTo.id);
    if (i == null) return;
    v.scrollToIndex(i, { align: "center" });
    // Re-trigger the flash animation on the row once it is rendered.
    const t = window.setTimeout(() => {
      const el = parent.current?.querySelector<HTMLElement>(`[data-seg="${scrollTo.id}"]`);
      if (el) {
        el.classList.remove("flash");
        void el.offsetWidth;
        el.classList.add("flash");
      }
    }, 80);
    return () => window.clearTimeout(t);
  }, [scrollTo, index, v]);

  const blockedCount = rejected.length;

  return (
    <section className="flex min-h-0 flex-col max-md:h-[65vh]" aria-label="Transcript">
      <header className="flex items-center justify-between gap-3 border-b border-line px-4 py-2.5">
        <h2 className="text-xs font-semibold tracking-wider text-ink-3 uppercase">Transcript</h2>
        <Tabs
          label="Transcript view"
          value={effectiveMode}
          onChange={onMode}
          items={
            hasRefinement
              ? [
                  { value: "raw", label: "Raw" },
                  { value: "refined", label: "Refined" },
                  { value: "diff", label: <>Diff{blockedCount + accepted.length ? <span className="ml-1 text-ink-3">{accepted.length + blockedCount}</span> : null}</> },
                ]
              : [{ value: "raw", label: "Raw" }]
          }
        />
      </header>
      <div ref={parent} className="min-h-0 flex-1 overflow-y-auto">
        <div style={{ height: v.getTotalSize(), position: "relative" }}>
          {v.getVirtualItems().map((item) => {
            const seg = raw[item.index];
            return (
              <div
                key={seg.id}
                data-index={item.index}
                data-seg={seg.id}
                ref={v.measureElement}
                style={{ position: "absolute", top: 0, left: 0, right: 0, transform: `translateY(${item.start}px)` }}
              >
                <Row
                  seg={seg}
                  pieces={pieces[item.index]}
                  mode={effectiveMode}
                  active={seg.id === activeId}
                  onSeek={onSeek}
                />
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
