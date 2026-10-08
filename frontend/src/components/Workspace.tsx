import { Fragment, useCallback, useEffect, useMemo, useState } from "react";
import { ApiError, type Api } from "../lib/api";
import type { TranscriptMode } from "../lib/annotate";
import { fmtTime } from "../lib/format";
import type { JobError, PartialRecord, Segment } from "../lib/types";
import { useAudio } from "../lib/useAudio";
import { DownloadMenu } from "./DownloadMenu";
import { PlayerBar, type Pin } from "./PlayerBar";
import { RecordPane, type RecordTab } from "./RecordPane";
import { Legend, Scorecard } from "./Scorecard";
import { TranscriptPane } from "./TranscriptPane";
import { Brand, Button, Icon, Tip } from "./ui";

interface Props {
  api: Api;
  jobId: string;
  record: PartialRecord;
  audioUrl: string | null;
  error?: JobError;
  /** Run the failed steps again (keeps everything that already finished). */
  onRetry?: () => Promise<void>;
  theme: "light" | "dark";
  onTheme: () => void;
  onNew: () => void;
}

/** Plays a moment slightly early so the listener hears it in context. */
const LEAD_IN_S = 2;

export function Workspace({ api, jobId, record: initialRecord, audioUrl, error, theme, onTheme, onNew, onRetry }: Props) {
  const [retrying, setRetrying] = useState(false);
  const [retryError, setRetryError] = useState<string | null>(null);
  const [record, setRecord] = useState<PartialRecord>(initialRecord);

  useEffect(() => {
    setRecord(initialRecord);
  }, [initialRecord]);

  const raw = record.raw_transcript ?? [];
  const audio = useAudio(audioUrl, record.meta.duration_s ?? raw.at(-1)?.end ?? 0);
  const [mode, setMode] = useState<TranscriptMode>(record.refinement ? "diff" : "raw");
  const [tab, setTab] = useState<RecordTab>("summary");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [scrollTo, setScrollTo] = useState<{ id: string; nonce: number } | null>(null);

  const segments = useMemo(() => new Map<string, Segment>(raw.map((s) => [s.id, s])), [raw]);

  const handleStartEdit = useCallback(() => {
    if (audio.playing) {
      audio.pause();
    }
  }, [audio]);

  const handleUpdateWord = useCallback((segmentId: string, wordIdx: number, newWord: string) => {
    const trimmed = newWord.trim();
    if (!trimmed) return;

    setRecord((prev) => {
      const updateSegList = (list?: Segment[]) => {
        if (!list) return list;
        return list.map((seg) => {
          if (seg.id !== segmentId) return seg;

          const newTokens = trimmed.split(/\s+/);
          const words = seg.words ? seg.words.map((w) => ({ ...w })) : undefined;
          let newText = seg.text;

          if (words && words.length > 0 && wordIdx >= 0 && wordIdx < words.length) {
            const origW = words[wordIdx];
            const start = origW.start;
            const end = origW.end;
            const dur = Math.max(0.15, end - start);
            const step = dur / newTokens.length;

            const createdWords = newTokens.map((w, idx) => ({
              w,
              start: +(start + idx * step).toFixed(2),
              end: +(start + (idx + 1) * step).toFixed(2),
              conf: origW.conf ?? 1.0,
            }));

            words.splice(wordIdx, 1, ...createdWords);

            const textTokens = seg.text.split(/(\s+)/);
            let wCount = 0;
            const nextTokens = textTokens.map((tok) => {
              if (/^\s+$/.test(tok)) return tok;
              if (wCount++ === wordIdx) return newTokens.join(" ");
              return tok;
            });
            newText = nextTokens.join("");
          } else {
            const textTokens = seg.text.split(/(\s+)/);
            let wCount = 0;
            const nextTokens = textTokens.map((tok) => {
              if (/^\s+$/.test(tok)) return tok;
              if (wCount++ === wordIdx) return newTokens.join(" ");
              return tok;
            });
            newText = nextTokens.join("");
          }

          return {
            ...seg,
            text: newText,
            words,
          };
        });
      };

      return {
        ...prev,
        raw_transcript: updateSegList(prev.raw_transcript),
        refined_transcript: updateSegList(prev.refined_transcript),
      };
    });
  }, []);

  const handleUpdateSentence = useCallback((segmentId: string, newSentence: string) => {
    const trimmed = newSentence.trim();
    if (!trimmed) return;

    setRecord((prev) => {
      const updateSegList = (list?: Segment[]) => {
        if (!list) return list;
        return list.map((seg) => {
          if (seg.id !== segmentId) return seg;

          const newTokens = trimmed.split(/\s+/);
          const dur = Math.max(0.5, seg.end - seg.start);
          const step = dur / newTokens.length;
          const words = newTokens.map((w, idx) => ({
            w,
            start: +(seg.start + idx * step).toFixed(2),
            end: +(seg.start + (idx + 1) * step).toFixed(2),
            conf: 1.0,
          }));

          return {
            ...seg,
            text: trimmed,
            words,
          };
        });
      };

      return {
        ...prev,
        raw_transcript: updateSegList(prev.raw_transcript),
        refined_transcript: updateSegList(prev.refined_transcript),
      };
    });
  }, []);


  const startOf = useCallback(
    (ids: string[]) => Math.min(...ids.map((i) => segments.get(i)?.start ?? Infinity)),
    [segments],
  );

  // Decisions and tasks in time order: timeline pins and J/K navigation.
  const pins: Pin[] = useMemo(() => {
    const out: Pin[] = [
      ...(record.decisions ?? []).map((d) => ({ id: d.id, kind: "decision" as const, label: d.decision, time: startOf(d.evidence_segment_ids) })),
      ...(record.action_items ?? []).map((a) => ({ id: a.id, kind: "task" as const, label: a.task, time: startOf(a.evidence_segment_ids) })),
    ];
    return out.filter((p) => isFinite(p.time)).sort((a, b) => a.time - b.time);
  }, [record, startOf]);

  const activeId = useMemo(() => {
    if (!audio.available || (!audio.playing && audio.time === 0)) return null;
    return raw.find((s) => audio.time >= s.start && audio.time < s.end)?.id ?? null;
  }, [raw, audio.time, audio.playing, audio.available]);

  const jump = useCallback(
    (ids: string[], itemId?: string) => {
      const first = [...ids].sort((a, b) => (segments.get(a)?.start ?? 0) - (segments.get(b)?.start ?? 0))[0];
      if (!first) return;
      setScrollTo({ id: first, nonce: Date.now() });
      if (itemId) setSelectedId(itemId);
      const seg = segments.get(first);
      if (seg && audio.available) audio.playFrom(seg.start - LEAD_IN_S);
    },
    [segments, audio],
  );

  const selectPin = useCallback(
    (id: string) => {
      const isDecision = record.decisions?.some((d) => d.id === id);
      setTab(isDecision ? "decisions" : "actions");
      const ids =
        record.decisions?.find((d) => d.id === id)?.evidence_segment_ids ??
        record.action_items?.find((a) => a.id === id)?.evidence_segment_ids ??
        [];
      jump(ids, id);
    },
    [record, jump],
  );

  // Keyboard: Space play/pause, J/K next/previous decision or task.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement;
      if (t.closest("input, textarea, select, [contenteditable=true]") || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key === " " && !t.closest("button, [role=slider]")) {
        e.preventDefault();
        audio.toggle();
      } else if ((e.key === "j" || e.key === "k") && pins.length) {
        const i = pins.findIndex((p) => p.id === selectedId);
        const next = e.key === "j" ? (i + 1) % pins.length : i <= 0 ? pins.length - 1 : i - 1;
        selectPin(pins[next].id);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [audio, pins, selectedId, selectPin]);

  const m = record.meta.models ?? {};
  // Slowest steps first, so the tooltip shows where the time went.
  const timings = Object.entries(record.meta.timings ?? {}).sort((a, b) => b[1] - a[1]);
  const totalSec = timings.reduce((t, [, s]) => t + s, 0);

  return (
    <div className="flex h-full flex-col">
      <audio ref={audio.ref} src={audioUrl ?? undefined} preload="metadata" />

      <header className="flex flex-wrap items-center gap-x-5 gap-y-2 border-b border-ink bg-surface px-4 py-2.5">
        <Brand compact />
        <div className="min-w-0 flex-1 border-l border-line pl-5">
          <div className="truncate text-[15px] font-bold tracking-[-0.01em]">{record.meta.title ?? record.meta.source_file ?? "Recording"}</div>
          <div className="truncate font-mono text-[10.5px] text-ink-3">
            {fmtTime(record.meta.duration_s ?? audio.duration)}
            {(m.stt || m.lm1 || m.lm2) && (
              <Tip
                className="ml-2 cursor-help"
                content={
                  <dl className="grid grid-cols-[auto_1fr] gap-x-3">
                    {m.stt && <><dt className="text-ink-3">Speech</dt><dd>{m.stt}</dd></>}
                    {m.stt_check && <><dt className="text-ink-3">Second opinion</dt><dd>{m.stt_check}</dd></>}
                    {m.diarization && <><dt className="text-ink-3">Speakers</dt><dd>{m.diarization}</dd></>}
                    {m.lm1 && <><dt className="text-ink-3">LM1 refiner</dt><dd>{m.lm1}</dd></>}
                    {m.lm2 && <><dt className="text-ink-3">LM2 documenter</dt><dd>{m.lm2}</dd></>}
                  </dl>
                }
              >
                · {[m.stt, m.lm1, m.lm2].filter(Boolean).join(" / ")}
              </Tip>
            )}
            {timings.length > 0 && (
              <Tip
                className="ml-2 cursor-help"
                content={
                  <dl className="grid grid-cols-[auto_auto] gap-x-3">
                    {timings.map(([stage, s]) => (
                      <Fragment key={stage}>
                        <dt className="text-ink-3">{stage.replace("_", " ")}</dt>
                        <dd className="text-right tabular-nums">{s.toFixed(1)} s</dd>
                      </Fragment>
                    ))}
                  </dl>
                }
              >
                · processed in {fmtTime(totalSec)}
              </Tip>
            )}
          </div>
        </div>
        <DownloadMenu api={api} jobId={jobId} record={record} />
        <Button variant="ghost" onClick={onTheme} aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}>
          {theme === "dark" ? <Icon.sun /> : <Icon.moon />}
        </Button>
        <Button onClick={onNew}>
          <Icon.plus /> New recording
        </Button>
      </header>

      {record.meta.warnings?.includes("W_NON_ENGLISH") && (
        <div role="status" className="border-b border-warn/30 bg-warn-soft px-4 py-2 text-sm text-warn">
          This recording may not be in English; results may be less accurate.
        </div>
      )}

      {error && record.summary == null && (
        <div role="alert" className="flex flex-wrap items-center gap-x-2 gap-y-1 border-b border-bad/30 bg-bad-soft px-4 py-2 text-sm text-bad">
          <Icon.alert /> {error.user_message}
          <span className="font-mono text-[11px] text-ink-3">{error.code}</span>
          {error.detail && <span className="min-w-0 basis-full font-mono text-[11px] break-words text-ink-2 sm:basis-auto">{error.detail}</span>}
          {retryError && <span className="text-xs">{retryError}</span>}
          {onRetry && (
            <Button
              size="sm"
              className="ml-auto"
              disabled={retrying}
              onClick={async () => {
                setRetrying(true);
                setRetryError(null);
                try {
                  await onRetry();
                } catch (e) {
                  setRetrying(false);
                  setRetryError(e instanceof ApiError ? e.jobError.user_message : "Retry failed.");
                }
              }}
            >
              {retrying ? "Starting…" : "Retry from this step"}
            </Button>
          )}
        </div>
      )}

      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-line bg-bg px-3 py-1.5">
        <Scorecard f={record.fidelity ?? {}} />
        <Legend />
      </div>

      <main className="grid min-h-0 flex-1 grid-cols-1 divide-ink/15 bg-surface max-md:overflow-y-auto md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)] md:divide-x">
        <TranscriptPane
          raw={raw}
          accepted={record.refinement?.accepted ?? []}
          rejected={record.refinement?.rejected ?? []}
          hasRefinement={!!record.refinement}
          mode={mode}
          onMode={setMode}
          activeId={activeId}
          currentTime={audio.time}
          isPlaying={audio.playing}
          scrollTo={scrollTo}
          onSeek={(t) => audio.available && audio.playFrom(t)}
          onUpdateWord={handleUpdateWord}
          onUpdateSentence={handleUpdateSentence}
          onStartEdit={handleStartEdit}
        />
        <RecordPane
          record={record}
          segments={segments}
          tab={tab}
          onTab={setTab}
          selectedId={selectedId}
          onJump={jump}
          error={error}
        />
      </main>

      <PlayerBar audio={audio} pins={pins} selectedId={selectedId} onPin={selectPin} />
      <div className="flex justify-between border-t border-line bg-surface px-4 py-1 text-[11px] text-ink-3">
        <span>{audio.available ? "Click any chip or timestamp to hear that moment." : "Audio playback is unavailable for this record."}</span>
        <span className="hidden sm:inline">
          <kbd className="font-mono">Space</kbd> play/pause · <kbd className="font-mono">J</kbd>/<kbd className="font-mono">K</kbd> next/previous item
        </span>
      </div>
    </div>
  );
}
