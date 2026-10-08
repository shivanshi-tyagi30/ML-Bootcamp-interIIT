import { useVirtualizer } from "@tanstack/react-virtual";
import { memo, useEffect, useMemo, useRef, useState } from "react";
import { REJECT_REASONS, annotate, type Piece, type TranscriptMode } from "../lib/annotate";
import { fmtTime } from "../lib/format";
import type { Edit, RejectedEdit, Segment } from "../lib/types";
import { Icon, Tabs, Tip, cx } from "./ui";

interface Props {
  raw: Segment[];
  accepted: Edit[];
  rejected: RejectedEdit[];
  hasRefinement: boolean;
  mode: TranscriptMode;
  onMode: (m: TranscriptMode) => void;
  activeId: string | null;
  currentTime?: number;
  isPlaying?: boolean;
  scrollTo: { id: string; nonce: number } | null;
  onSeek: (t: number) => void;
  onUpdateWord?: (segmentId: string, wordIdx: number, newWord: string) => void;
  onUpdateSentence?: (segmentId: string, newSentence: string) => void;
  onStartEdit?: () => void;
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

function EditableWord({
  word,
  wordIdx,
  segId,
  onSeek,
  onSave,
  onStartEdit,
  children,
  className,
}: {
  word: string;
  wordIdx: number;
  segId: string;
  onSeek?: () => void;
  onSave?: (segmentId: string, wordIdx: number, newWord: string) => void;
  onStartEdit?: () => void;
  children?: React.ReactNode;
  className?: string;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(word);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    setDraft(word);
  }, [word]);

  useEffect(() => {
    if (editing) {
      inputRef.current?.focus();
      inputRef.current?.select();
    }
  }, [editing]);

  const commit = () => {
    const trimmed = draft.trim();
    if (trimmed && trimmed !== word) {
      onSave?.(segId, wordIdx, trimmed);
    } else {
      setDraft(word);
    }
    setEditing(false);
  };

  const cancel = () => {
    setDraft(word);
    setEditing(false);
  };

  if (editing) {
    return (
      <input
        ref={inputRef}
        type="text"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onKeyDown={(e) => {
          e.stopPropagation();
          if (e.key === "Enter") {
            e.preventDefault();
            commit();
          } else if (e.key === "Escape") {
            e.preventDefault();
            cancel();
          }
        }}
        onBlur={commit}
        onClick={(e) => e.stopPropagation()}
        onDoubleClick={(e) => e.stopPropagation()}
        style={{ minWidth: "10ch", width: `${Math.max(10, draft.length + 2)}ch`, maxWidth: "100%" }}
        className="inline-block text-[15px] font-semibold text-ink bg-surface border-2 border-accent rounded px-2 py-0.5 shadow-xs outline-none ring-2 ring-accent/30 z-30 relative -my-0.5 align-baseline text-accent-deep"
      />
    );
  }

  // Click plays from the word; editing has its own small pencil button that shows on hover, so a click
  // never starts an edit by accident. Hover popups on edited or disputed words keep working.
  return (
    <span className="group/word relative inline-block">
      <span onClick={() => onSeek?.()} title={onSeek ? "Click to play from here" : undefined} className={className}>
        {children ?? word}
      </span>
      {onSave && (
        <button
          type="button"
          aria-label={`Edit "${word}"`}
          title="Edit this word"
          onMouseDown={(e) => e.preventDefault()}
          onClick={(e) => {
            e.stopPropagation();
            onStartEdit?.();
            setDraft(word);
            setEditing(true);
          }}
          className="absolute -bottom-2 -right-2 z-20 flex h-4 w-4 items-center justify-center rounded-full border border-ink/20 bg-surface text-ink-2 opacity-0 shadow-xs transition-opacity hover:border-accent hover:text-accent-deep focus:opacity-100 group-hover/word:opacity-100"
        >
          <Icon.pencil className="h-2.5 w-2.5" />
        </button>
      )}
    </span>
  );
}

function SegmentContent({
  seg,
  pieces,
  mode,
  active,
  currentTime,
  isPlaying,
  onSeek,
  onUpdateWord,
  onStartEdit,
}: {
  seg: Segment;
  pieces: Piece[];
  mode: TranscriptMode;
  active: boolean;
  currentTime: number;
  isPlaying: boolean;
  onSeek: (t: number) => void;
  onUpdateWord?: (segmentId: string, wordIdx: number, newWord: string) => void;
  onStartEdit?: () => void;
}) {
  const words = seg.words;
  if (!words || !words.length) {
    let wordCursor = 0;
    return (
      <p className="text-[15px] leading-relaxed">
        {pieces.map((p, pIdx) => {
          if (p.kind === "text") {
            const tokens = p.text.split(/(\s+)/);
            return (
              <span key={pIdx}>
                {tokens.map((tok, tIdx) => {
                  if (!tok) return null;
                  if (/^\s+$/.test(tok)) return <span key={tIdx}>{tok}</span>;
                  const wIdx = wordCursor++;
                  return (
                    <EditableWord
                      key={tIdx}
                      word={tok}
                      wordIdx={wIdx}
                      segId={seg.id}
                      onSave={onUpdateWord}
                      onStartEdit={onStartEdit}
                      className="inline-block scale-100 cursor-pointer hover:text-accent-deep"
                    />
                  );
                })}
              </span>
            );
          }
          const wIdx = wordCursor++;
          return (
            <EditableWord
              key={pIdx}
              word={p.kind === "disputed" ? p.text : p.kind === "accepted" ? p.edit.replacement : p.edit.original}
              wordIdx={wIdx}
              segId={seg.id}
              onSave={onUpdateWord}
              onStartEdit={onStartEdit}
              className="inline-block scale-100 cursor-pointer"
            >
              <PieceView p={p} mode={mode} />
            </EditableWord>
          );
        })}
      </p>
    );
  }

  let activeWordIdx = -1;
  if (active && isPlaying) {
    activeWordIdx = words.findIndex((w, i, arr) => {
      const nextStart = arr[i + 1]?.start ?? (w.end + 0.35);
      return currentTime >= w.start && currentTime < Math.max(w.end, nextStart);
    });
  }

  let wordCursor = 0;

  return (
    <p className="text-[15px] leading-relaxed">
      {pieces.map((p, pIdx) => {
        if (p.kind === "text") {
          const tokens = p.text.split(/(\s+)/);
          return (
            <span key={pIdx}>
              {tokens.map((tok, tIdx) => {
                if (!tok) return null;
                if (/^\s+$/.test(tok)) {
                  return <span key={tIdx}>{tok}</span>;
                }
                const wIdx = wordCursor++;
                const w = words[wIdx];
                const isCurrent = wIdx === activeWordIdx;
                return (
                  <EditableWord
                    key={tIdx}
                    word={tok}
                    wordIdx={wIdx}
                    segId={seg.id}
                    onSeek={w ? () => onSeek(w.start) : undefined}
                    onSave={onUpdateWord}
                    onStartEdit={onStartEdit}
                    className={cx(
                      "inline-block transition-all duration-150 ease-out origin-bottom",
                      isCurrent
                        ? "scale-[1.06] -translate-y-[1px] font-semibold text-accent-deep bg-accent-soft/80 px-0.5 rounded shadow-xs z-10 relative"
                        : "scale-100",
                      w && "cursor-pointer hover:text-accent-deep",
                    )}
                  />
                );
              })}
            </span>
          );
        }

        const pieceWordCount = p.kind === "disputed" ? 1 : Math.max(1, p.edit.original.trim().split(/\s+/).length);
        const startWIdx = wordCursor;
        const endWIdx = wordCursor + pieceWordCount;
        wordCursor += pieceWordCount;

        const isCurrent = activeWordIdx >= startWIdx && activeWordIdx < endWIdx;
        const pieceStart = words[startWIdx]?.start ?? seg.start;
        const pieceWord = p.kind === "disputed" ? p.text : p.kind === "accepted" ? p.edit.replacement : p.edit.original;

        return (
          <EditableWord
            key={pIdx}
            word={pieceWord}
            wordIdx={startWIdx}
            segId={seg.id}
            onSeek={() => onSeek(pieceStart)}
            onSave={onUpdateWord}
            onStartEdit={onStartEdit}
            className={cx(
              "inline-block transition-all duration-150 ease-out origin-bottom",
              isCurrent
                ? "scale-[1.06] -translate-y-[1px] font-semibold shadow-xs z-10 relative ring-1 ring-accent/40 rounded px-0.5"
                : "scale-100",
              "cursor-pointer",
            )}
          >
            <PieceView p={p} mode={mode} />
          </EditableWord>
        );
      })}
    </p>
  );
}

const Row = memo(function Row({
  seg,
  pieces,
  mode,
  active,
  currentTime,
  isPlaying,
  onSeek,
  onUpdateWord,
  onUpdateSentence,
  onStartEdit,
}: {
  seg: Segment;
  pieces: Piece[];
  mode: TranscriptMode;
  active: boolean;
  currentTime: number;
  isPlaying: boolean;
  onSeek: (t: number) => void;
  onUpdateWord?: (segmentId: string, wordIdx: number, newWord: string) => void;
  onUpdateSentence?: (segmentId: string, newSentence: string) => void;
  onStartEdit?: () => void;
}) {
  const [editingSentence, setEditingSentence] = useState(false);
  const [sentenceDraft, setSentenceDraft] = useState(seg.text);

  useEffect(() => {
    setSentenceDraft(seg.text);
  }, [seg.text]);

  const commitSentence = () => {
    const trimmed = sentenceDraft.trim();
    if (trimmed && trimmed !== seg.text) {
      onUpdateSentence?.(seg.id, trimmed);
    } else {
      setSentenceDraft(seg.text);
    }
    setEditingSentence(false);
  };

  return (
    <div
      className={cx(
        "group grid grid-cols-[3.5rem_1fr] gap-x-3 border-l-2 px-4 py-2.5 transition-all duration-300 ease-out rounded-r-md",
        active
          ? "border-accent bg-accent-soft/40 shadow-sm translate-x-1"
          : "border-transparent hover:bg-raised/40",
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

      {editingSentence ? (
        <div className="flex-1">
          <div className="mb-1 flex items-center justify-between text-xs font-medium text-ink-2">
            <span>{seg.speaker}</span>
            <span className="text-[10.5px] text-ink-3">Press Enter to save · Esc to cancel</span>
          </div>
          <textarea
            autoFocus
            value={sentenceDraft}
            onChange={(e) => setSentenceDraft(e.target.value)}
            onKeyDown={(e) => {
              e.stopPropagation();
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                commitSentence();
              } else if (e.key === "Escape") {
                e.preventDefault();
                setSentenceDraft(seg.text);
                setEditingSentence(false);
              }
            }}
            onBlur={commitSentence}
            rows={2}
            className="w-full text-[15px] leading-relaxed font-medium text-ink bg-surface border-2 border-accent rounded p-2 shadow-xs outline-none ring-2 ring-accent/30 resize-none"
          />
        </div>
      ) : (
        <div>
          <div className="mb-0.5 flex items-center justify-between text-xs font-medium text-ink-2">
            {seg.speaker && <span>{seg.speaker}</span>}
            <button
              onClick={() => {
                onStartEdit?.();
                setSentenceDraft(seg.text);
                setEditingSentence(true);
              }}
              className="opacity-0 group-hover:opacity-100 hover:text-accent-deep text-[11px] text-ink-3 transition-opacity ml-auto"
              title="Edit entire sentence"
            >
              ✏ Edit line
            </button>
          </div>
          <SegmentContent
            seg={seg}
            pieces={pieces}
            mode={mode}
            active={active}
            currentTime={currentTime}
            isPlaying={isPlaying}
            onSeek={onSeek}
            onUpdateWord={onUpdateWord}
            onStartEdit={onStartEdit}
          />
        </div>
      )}
    </div>
  );
});

export function TranscriptPane({
  raw,
  accepted,
  rejected,
  hasRefinement,
  mode,
  onMode,
  activeId,
  currentTime = 0,
  isPlaying = false,
  scrollTo,
  onSeek,
  onUpdateWord,
  onUpdateSentence,
  onStartEdit,
}: Props) {
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

  // Smoothly track and scroll to the currently spoken sentence as audio plays
  useEffect(() => {
    if (!isPlaying || !activeId) return;
    const i = index.get(activeId);
    if (i == null) return;
    v.scrollToIndex(i, { align: "center", behavior: "smooth" });
  }, [activeId, isPlaying, index, v]);

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
        <div className="flex items-center gap-2">
          <h2 className="text-xs font-semibold tracking-wider text-ink-3 uppercase">Transcript</h2>
          <span className="hidden sm:inline-block rounded bg-raised px-1.5 py-0.5 text-[10.5px] text-ink-3">
            Hover a word, then ✎ to edit
          </span>
        </div>
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
      <div ref={parent} className="min-h-0 flex-1 overflow-y-auto scroll-smooth">
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
                  currentTime={currentTime}
                  isPlaying={isPlaying}
                  onSeek={onSeek}
                  onUpdateWord={onUpdateWord}
                  onUpdateSentence={onUpdateSentence}
                  onStartEdit={onStartEdit}
                />
              </div>
            );
          })}
        </div>
      </div>
    </section>
  );
}
