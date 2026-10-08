import { useEffect, useRef, useState } from "react";

// Interactive evidence chain demonstration: audio waveform → transcript word → action item.
// Demonstrates word-level audio sync and zero-hallucination decision extraction.

const WORDS = [
  { text: "I'll", start: 0.0, end: 0.5 },
  { text: "update", start: 0.5, end: 1.1 },
  { text: "the", start: 1.1, end: 1.4 },
  { text: "dashboards", start: 1.4, end: 2.6 },
  { text: "by", start: 2.6, end: 3.1 },
  { text: "Monday", start: 3.1, end: 4.1 },
  { text: "before", start: 4.1, end: 4.7 },
  { text: "the", start: 4.7, end: 5.0 },
  { text: "deploy.", start: 5.0, end: 6.0 },
];

const BARS = [
  8, 14, 22, 12, 30, 20, 40, 26, 48, 34, 22, 54, 42, 66, 50, 36, 72, 58, 84, 64, 46, 60, 36, 50, 26, 40, 20, 28, 14, 18, 10,
];

const DURATION = 6.0;
const WAVEFORM_WIDTH = 180;

export function HeroIllustration() {
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0); // Start at beginning
  const animFrame = useRef<number | null>(null);
  const lastTime = useRef<number | null>(null);

  useEffect(() => {
    if (!isPlaying) {
      if (animFrame.current) cancelAnimationFrame(animFrame.current);
      lastTime.current = null;
      return;
    }

    const step = (now: number) => {
      if (lastTime.current !== null) {
        const delta = (now - lastTime.current) / 1000;
        let reachedEnd = false;
        setCurrentTime((prev) => {
          const next = prev + delta;
          if (next >= DURATION) {
            reachedEnd = true;
            return DURATION;
          }
          return next;
        });

        if (reachedEnd) {
          setIsPlaying(false);
          return; // Stop after running once
        }
      }
      lastTime.current = now;
      animFrame.current = requestAnimationFrame(step);
    };

    animFrame.current = requestAnimationFrame(step);
    return () => {
      if (animFrame.current) cancelAnimationFrame(animFrame.current);
    };
  }, [isPlaying]);

  const hasEnded = currentTime >= DURATION;

  const togglePlay = () => {
    if (isPlaying) {
      setIsPlaying(false);
    } else {
      if (hasEnded) {
        setCurrentTime(0);
      }
      setIsPlaying(true);
    }
  };

  const progress = Math.min(1, Math.max(0, currentTime / DURATION));
  const playheadX = Math.round(progress * (WAVEFORM_WIDTH - 6)) + 3;
  const currentBarIndex = Math.min(BARS.length - 1, Math.floor(progress * BARS.length));

  // Determine active word
  const activeWordIndex = WORDS.findIndex((w) => currentTime >= w.start && currentTime < w.end);
  const isActionItemActive = currentTime >= 2.6 && currentTime <= 4.5;

  const seek = (time: number) => {
    setCurrentTime(Math.min(DURATION, Math.max(0, time)));
  };

  const formatSeconds = (sec: number) => {
    const total = 134 + sec; // base 02:14 (134s)
    const m = Math.floor(total / 60);
    const s = Math.floor(total % 60);
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  };

  return (
    <div className="relative select-none" aria-label="Interactive demo of Trace audio sync and action item extraction">
      
      {/* Top Demo Interactive Header Bar */}
      <div className="mb-3 flex items-center justify-between border-b border-ink/15 pb-2 text-[10.5px] font-mono">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={togglePlay}
            className="flex items-center gap-1.5 bg-accent px-2.5 py-1 font-bold text-accent-ink transition hover:bg-ink hover:text-accent shadow-sm"
          >
            {isPlaying ? (
              <>
                <svg className="size-3 fill-current" viewBox="0 0 24 24">
                  <path d="M6 19h4V5H6v14zm8-14v14h4V5h-4z" />
                </svg>
                <span>PAUSE DEMO</span>
              </>
            ) : hasEnded ? (
              <>
                <svg className="size-3 fill-current" viewBox="0 0 24 24">
                  <path d="M12 5V1L7 6l5 5V7c3.31 0 6 2.69 6 6s-2.69 6-6 6-6-2.69-6-6H4c0 4.42 3.58 8 8 8s8-3.58 8-8-3.58-8-8-8z" />
                </svg>
                <span>REPLAY DEMO</span>
              </>
            ) : (
              <>
                <svg className="size-3 fill-current" viewBox="0 0 24 24">
                  <path d="M8 5v14l11-7z" />
                </svg>
                <span>PLAY SAMPLE (6s)</span>
              </>
            )}
          </button>
          <span className="font-semibold text-ink-2">{formatSeconds(currentTime)}</span>
        </div>

        <span className="text-ink-3 hidden sm:inline">Click word or bar to scrub</span>
      </div>

      <div className="relative h-[255px] w-[560px]">
        {/* Interactive Waveform */}
        <div
          role="slider"
          aria-label="Audio waveform scrubber"
          aria-valuenow={Math.round(currentTime * 10) / 10}
          aria-valuemin={0}
          aria-valuemax={DURATION}
          tabIndex={0}
          className="group absolute top-[132px] left-0 flex -translate-y-1/2 cursor-pointer items-center gap-[3px] py-4"
          onClick={(e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            seek((clickX / rect.width) * DURATION);
          }}
        >
          {BARS.map((h, i) => {
            const isNearPlayhead = Math.abs(i - currentBarIndex) <= 2;
            const isPast = i <= currentBarIndex;
            return (
              <span
                key={i}
                className={`w-[3px] rounded-full transition-all duration-75 ${
                  isNearPlayhead
                    ? "bg-accent scale-y-110"
                    : isPast
                    ? "bg-ink opacity-90"
                    : "bg-ink/30"
                }`}
                style={{ height: `${h * 0.8}px` }}
              />
            );
          })}
        </div>

        {/* Playhead */}
        <div
          className="pointer-events-none absolute top-[78px] transition-all duration-75"
          style={{ left: `${playheadX}px` }}
        >
          <span className="block h-[112px] border-l border-dashed border-accent" />
          <span className="absolute top-[120px] left-0 -translate-x-1/2 bg-ink px-1.5 py-0.5 font-mono text-[9px] font-bold text-surface shadow">
            {formatSeconds(currentTime)}
          </span>
        </div>

        {/* Dynamic Curved SVG connecting lines */}
        <svg className="pointer-events-none absolute inset-0 overflow-visible" width="560" height="260">
          <path
            d={`M${playheadX},78 C${playheadX + 35},78 136,108 158,108`}
            fill="none"
            className={`transition-colors duration-200 ${
              isActionItemActive ? "stroke-accent" : "stroke-ink/40"
            }`}
            strokeWidth={isActionItemActive ? 2 : 1.2}
            strokeDasharray={isPlaying ? "4 3" : undefined}
          />
          <path
            d="M362,108 C384,108 376,146 394,146"
            fill="none"
            className={`transition-colors duration-200 ${
              isActionItemActive ? "stroke-accent" : "stroke-ink/30"
            }`}
            strokeWidth={isActionItemActive ? 2 : 1.2}
            strokeDasharray={isPlaying && isActionItemActive ? "4 3" : undefined}
          />
        </svg>

        {/* Interactive Transcript Card */}
        <div className="absolute top-[4px] left-[158px] h-[218px] w-[204px] border border-ink/15 bg-surface p-3 shadow-[6px_6px_0_rgba(17,17,17,0.06)] transition-all">
          <div className="flex items-center justify-between font-mono text-[8.5px] tracking-[0.16em] text-ink-3">
            <span>TRANSCRIPT</span>
            <span className="text-[7.5px] text-accent font-bold">SYNCED</span>
          </div>

          {/* Decorative past lines */}
          <div className="mt-2 space-y-1.5">
            <div className="h-1 w-28 bg-ink/10 rounded-full" />
            <div className="h-1 w-36 bg-ink/10 rounded-full" />
          </div>

          {/* Active Spoken Segment */}
          <div className="mt-3 rounded border border-ink/10 bg-raised/40 p-2">
            <div className="flex items-center justify-between font-mono text-[8px] text-ink-3">
              <span className="font-semibold text-ink">PRIYA</span>
              <span>02:14</span>
            </div>
            <div className="mt-1 text-[11px] leading-[1.4] text-ink">
              {WORDS.map((w, idx) => {
                const isActive = idx === activeWordIndex;
                const isPast = currentTime >= w.end;
                return (
                  <button
                    key={idx}
                    type="button"
                    onClick={() => {
                      seek(w.start);
                      setIsPlaying(true);
                    }}
                    className={`cursor-pointer rounded px-0.5 transition-colors ${
                      isActive
                        ? "bg-accent text-accent-ink font-bold scale-105 inline-block shadow-sm"
                        : isPast
                        ? "text-ink font-medium hover:text-accent"
                        : "text-ink-3 hover:text-ink"
                    }`}
                  >
                    {w.text}{" "}
                  </button>
                );
              })}
            </div>
          </div>

          {/* Decorative upcoming lines */}
          <div className="mt-3 space-y-1.5">
            <div className="h-1 w-32 bg-ink/10 rounded-full" />
            <div className="h-1 w-20 bg-ink/10 rounded-full" />
          </div>
        </div>

        {/* Interactive Action Item Card */}
        <button
          type="button"
          onClick={() => {
            seek(3.1);
            setIsPlaying(true);
          }}
          className={`absolute top-[28px] left-[394px] h-[146px] w-[166px] text-left border bg-surface p-3 transition-all ${
            isActionItemActive
              ? "border-accent ring-2 ring-accent/30 shadow-[6px_6px_0_var(--color-accent)] scale-[1.02]"
              : "border-ink/15 shadow-[6px_6px_0_rgba(17,17,17,0.06)] hover:border-ink/40"
          }`}
        >
          <div className="flex items-center justify-between font-mono text-[8px] tracking-[0.14em] text-ink-3">
            <span>ACTION ITEM</span>
            <span className={`text-[7px] font-bold ${isActionItemActive ? "text-accent" : "text-emerald-600"}`}>
              VERIFIED
            </span>
          </div>

          <div className="mt-1 text-[12px] font-bold leading-tight text-ink">
            Update the dashboards
          </div>

          <dl className="mt-2 grid grid-cols-[40px_1fr] gap-y-1 text-[9.5px]">
            <dt className="text-ink-3">Owner</dt>
            <dd className="font-semibold text-ink">Priya</dd>
            <dt className="text-ink-3">Due</dt>
            <dd className="font-semibold text-accent">Monday</dd>
          </dl>

          <div className="mt-3 flex items-center gap-1 bg-ink px-1.5 py-0.5 font-mono text-[8px] text-surface w-fit rounded-[2px]">
            <svg viewBox="0 0 10 10" className="size-1.5 fill-accent">
              <path d="M2 1v8l7-4z" />
            </svg>
            <span>AUDIO PROOF · 02:14</span>
          </div>
        </button>

        {/* Second mini unassigned item (demonstrating zero hallucination) */}
        <div className="absolute top-[186px] left-[394px] h-[64px] w-[166px] border border-ink/15 bg-surface p-2.5 shadow-[6px_6px_0_rgba(17,17,17,0.06)]">
          <div className="text-[10.5px] font-semibold text-ink">Clean up the test data</div>
          <div className="mt-1 flex items-center justify-between text-[8.5px]">
            <span className="text-ink-3 font-mono">Owner: None named</span>
            <span className="rounded border border-dashed border-ink/30 px-1 py-0.5 text-ink-3 font-mono">
              Unassigned
            </span>
          </div>
        </div>

        {/* Interactive Nodes */}
        <span
          className="absolute size-2.5 -translate-x-1/2 -translate-y-1/2 border-2 border-ink bg-accent transition-transform"
          style={{ left: `${playheadX}px`, top: "78px" }}
        />
        <span
          className="absolute size-2.5 -translate-x-1/2 -translate-y-1/2 border-2 border-ink bg-accent"
          style={{ left: "158px", top: "108px" }}
        />
        <span
          className={`absolute size-2.5 -translate-x-1/2 -translate-y-1/2 border-2 border-ink transition-colors ${
            isActionItemActive ? "bg-accent scale-125" : "bg-ink"
          }`}
          style={{ left: "394px", top: "146px" }}
        />
      </div>
    </div>
  );
}
