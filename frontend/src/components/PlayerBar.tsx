import { useRef } from "react";
import { fmtTime } from "../lib/format";
import type { AudioControls } from "../lib/useAudio";
import { Icon, Tip, cx } from "./ui";

export interface Pin {
  id: string;
  kind: "decision" | "task";
  label: string;
  time: number;
}

const RATES = [0.75, 1, 1.25, 1.5, 2];

/** Bottom bar: play/pause, timeline with decision/task pins, speed (plan 11.1). */
export function PlayerBar({
  audio,
  pins,
  selectedId,
  onPin,
}: {
  audio: AudioControls;
  pins: Pin[];
  selectedId: string | null;
  onPin: (id: string) => void;
}) {
  const track = useRef<HTMLDivElement>(null);
  const dur = audio.duration || 1;
  const progress = Math.min(100, (audio.time / dur) * 100);

  const seekFromEvent = (clientX: number) => {
    const r = track.current?.getBoundingClientRect();
    if (!r) return;
    audio.seek(((clientX - r.left) / r.width) * dur);
  };

  return (
    <div className="flex items-center gap-3 border-t border-line bg-surface px-4 py-2.5">
      <button
        onClick={audio.toggle}
        disabled={!audio.available}
        aria-label={audio.playing ? "Pause" : "Play"}
        title={audio.available ? "Play / pause (Space)" : "No audio loaded"}
        className="grid size-9 shrink-0 place-items-center rounded-full bg-accent text-accent-ink hover:opacity-90 disabled:opacity-40"
      >
        {audio.playing ? <Icon.pause /> : <Icon.play className="translate-x-px" />}
      </button>
      <div className="w-24 shrink-0 font-mono text-xs text-ink-2 tabular-nums">
        {fmtTime(audio.time)} / {fmtTime(audio.duration)}
      </div>

      <div className="relative flex-1 py-3">
        <div
          ref={track}
          role="slider"
          tabIndex={audio.available ? 0 : -1}
          aria-label="Seek"
          aria-valuemin={0}
          aria-valuemax={Math.round(dur)}
          aria-valuenow={Math.round(audio.time)}
          aria-valuetext={fmtTime(audio.time)}
          onClick={(e) => seekFromEvent(e.clientX)}
          onKeyDown={(e) => {
            if (e.key === "ArrowRight") audio.seek(audio.time + 5);
            if (e.key === "ArrowLeft") audio.seek(audio.time - 5);
          }}
          className={cx("relative h-2 rounded-full bg-raised", audio.available ? "cursor-pointer" : "cursor-default")}
        >
          <div className="absolute inset-y-0 left-0 rounded-full bg-accent/35" style={{ width: `${progress}%` }} />
          <div
            className="absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-accent shadow"
            style={{ left: `${progress}%` }}
          />
        </div>
        {pins.map((p) => (
          <Tip
            key={p.id}
            focusable={false}
            content={
              <>
                <span className="font-mono text-ink-3">{p.id}</span> {p.label}{" "}
                <span className="text-ink-3">· {fmtTime(p.time)}</span>
              </>
            }
          >
            <button
              onClick={() => onPin(p.id)}
              aria-label={`${p.kind === "decision" ? "Decision" : "Task"} ${p.id}: ${p.label}`}
              style={{ left: `${(p.time / dur) * 100}%` }}
              className={cx(
                "absolute top-2.5 -translate-x-1/2 border-2 border-surface",
                p.kind === "decision" ? "size-3 rotate-45 rounded-[2px]" : "size-3 rounded-full",
                selectedId === p.id ? "bg-ink" : p.kind === "decision" ? "bg-accent" : "bg-ok",
              )}
            />
          </Tip>
        ))}
      </div>

      <div className="hidden items-center gap-3 text-[11px] text-ink-3 lg:flex">
        <span className="flex items-center gap-1"><span className="size-2 rotate-45 bg-accent" /> decision</span>
        <span className="flex items-center gap-1"><span className="size-2 rounded-full bg-ok" /> task</span>
      </div>

      <select
        value={audio.rate}
        onChange={(e) => audio.setRate(Number(e.target.value))}
        aria-label="Playback speed"
        className="h-7 rounded-[2px] border border-line bg-surface px-1.5 text-xs"
      >
        {RATES.map((r) => (
          <option key={r} value={r}>
            {r}×
          </option>
        ))}
      </select>
    </div>
  );
}
