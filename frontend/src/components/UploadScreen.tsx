import { useEffect, useRef, useState } from "react";
import { ACCEPTED_EXTENSIONS, MAX_UPLOAD_MB, precheckFile } from "../lib/errors";
import type { JobError } from "../lib/types";
import { BottomLeftBlob, TopRightBlob } from "./landing/Blobs";
import { DatePill } from "./landing/DatePill";
import { HeroIllustration } from "./landing/HeroIllustration";
import { cx } from "./ui";

interface Props {
  error: JobError | null;
  busy: boolean;
  mock: boolean;
  onStart: (file: File, glossary: string) => void;
  onSample: () => void;
  onClearError: () => void;
}

// The layout is drawn on a fixed 1078 × 606 canvas and scaled to the window,
// so wide screens keep the proportions of the design. Narrow screens stack.
const STAGE_W = 1078;
const STAGE_H = 606;
const WIDE_MIN = 900;

// Glass card outline: rounded rectangle with the top-right corner cut off.
const CARD_PATH =
  "M36,0 H198 Q212,0 222,10 L330,118 Q340,128 340,142 V334 A36,36 0 0 1 304,370 H36 A36,36 0 0 1 0,334 V36 A36,36 0 0 1 36,0 Z";

function useStage() {
  const get = () => {
    const w = window.innerWidth;
    const h = window.innerHeight;
    const scale = Math.min(w / STAGE_W, h / STAGE_H);
    return { wide: w >= WIDE_MIN, scale, x: (w - STAGE_W * scale) / 2, y: (h - STAGE_H * scale) / 2 };
  };
  const [s, setS] = useState(get);
  useEffect(() => {
    const on = () => setS(get());
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  return s;
}

function CloudIcon() {
  return (
    <svg viewBox="0 0 64 48" className="h-[38px] w-[52px] drop-shadow-[0_2px_8px_rgba(80,150,255,0.6)]" aria-hidden>
      <defs>
        <linearGradient id="cloud-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#bfe0ff" />
          <stop offset="1" stopColor="#5c9dff" />
        </linearGradient>
      </defs>
      <path
        d="M17 42h31a12 12 0 0 0 1.6-23.9A16 16 0 0 0 19 15.5 13.3 13.3 0 0 0 17 42z"
        fill="url(#cloud-fill)"
        stroke="#2f6fe8"
        strokeWidth="2.2"
        strokeLinejoin="round"
      />
      <path d="M32 44V25m0 0-6 6m6-6 6 6" fill="none" stroke="#1d56d8" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export function UploadScreen({ error, busy, mock, onStart, onSample, onClearError }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [glossary, setGlossary] = useState("");
  const [showGlossary, setShowGlossary] = useState(false);
  const [drag, setDrag] = useState(false);
  const [localError, setLocalError] = useState<JobError | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const stage = useStage();
  const shown = localError ?? error;
  const wide = stage.wide;

  const choose = (f: File | undefined) => {
    if (!f) return;
    onClearError();
    setLocalError(precheckFile(f));
    setFile(f);
  };

  const retry = () => {
    setFile(null);
    setLocalError(null);
    onClearError();
    if (input.current) input.current.value = "";
    input.current?.click();
  };

  const card = (
    <div className={cx("relative h-[370px] w-[340px] shrink-0", wide && "absolute top-[26px] left-[62px]")}>
      {/* glass body clipped to the card shape */}
      <div
        className="absolute inset-0"
        style={{
          clipPath: `path("${CARD_PATH}")`,
          background: "linear-gradient(160deg, rgba(255,255,255,0.34), rgba(255,255,255,0.08) 55%, rgba(255,255,255,0.12))",
          backdropFilter: "blur(20px) saturate(140%)",
          WebkitBackdropFilter: "blur(20px) saturate(140%)",
        }}
      />
      <svg className="pointer-events-none absolute inset-0" width="340" height="370" aria-hidden>
        <defs>
          <linearGradient id="card-rim" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0" stopColor="#fff" stopOpacity="0.85" />
            <stop offset="0.5" stopColor="#fff" stopOpacity="0.25" />
            <stop offset="1" stopColor="#fff" stopOpacity="0.5" />
          </linearGradient>
        </defs>
        <path d={CARD_PATH} fill="none" stroke="url(#card-rim)" strokeWidth="1.5" />
      </svg>

      <div className="absolute top-[47px] left-[36px] font-mono text-[11px] font-medium tracking-[0.14em] text-white/90">
        AI MEETING ASSISTANT
      </div>
      <h1 className="absolute top-[104px] left-[24px] text-[60px] leading-none font-extrabold tracking-[-0.045em] text-white [text-shadow:0_2px_20px_rgba(40,90,220,0.35)]">
        Trace
      </h1>

      <label
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          choose(e.dataTransfer.files[0]);
        }}
        className={cx(
          "absolute top-[188px] left-[19px] flex h-[162px] w-[301px] cursor-pointer flex-col rounded-[18px] border-2 border-dashed px-[16px] pt-[22px] pb-[16px] transition-colors",
          drag ? "border-white bg-white/15" : "border-white/80 hover:bg-white/[0.07]",
        )}
      >
        <input
          ref={input}
          type="file"
          className="sr-only"
          accept={ACCEPTED_EXTENSIONS.map((e) => "." + e).join(",") + ",audio/*"}
          onChange={(e) => choose(e.target.files?.[0])}
        />
        <div className="flex justify-center">
          <CloudIcon />
        </div>
        <div className="mt-auto font-mono text-[15px] leading-[1.35] text-white uppercase">
          {file ? (
            <>
              <span className="block truncate normal-case">{file.name}</span>
              <span className="text-[11px] text-white/70">
                {(file.size / 1024 / 1024).toFixed(1)} MB · CLICK TO CHANGE
              </span>
            </>
          ) : (
            "Drop an audio file/browse documents"
          )}
        </div>
      </label>
    </div>
  );

  const actions = (
    <div className={cx("flex flex-col gap-3", wide ? "absolute top-[459px] left-[62px] w-[340px] items-center" : "mt-8 w-[340px] items-center")}>
      <div className="flex items-center gap-4">
        <button
          disabled={!!localError || busy}
          // With no file yet, the button opens the file picker instead.
          onClick={() => (file ? onStart(file, glossary) : input.current?.click())}
          title={!file ? `Choose an audio file (${ACCEPTED_EXTENSIONS.join(", ").toUpperCase()}, up to ${MAX_UPLOAD_MB} MB)` : undefined}
          className="h-[36px] rounded-[3px] bg-[#0b6cf2] px-[14px] font-mono text-[16px] font-medium tracking-[0.02em] text-white shadow-[0_6px_24px_rgba(11,108,242,0.45)] transition hover:bg-[#2a80ff] disabled:cursor-not-allowed disabled:bg-[#0b6cf2]/70 disabled:text-white/75"
        >
          {busy ? "UPLOADING…" : "START PROCESSING"}
        </button>
      </div>

      <div className="flex items-center gap-4 font-mono text-[11px] tracking-[0.06em] text-white/55">
        <button className="hover:text-white" onClick={() => setShowGlossary((s) => !s)} aria-expanded={showGlossary}>
          {showGlossary ? "− EXPECTED TERMS" : "+ EXPECTED TERMS"}
        </button>
        {mock && (
          <button className="hover:text-white" onClick={onSample}>
            OPEN SAMPLE MEETING
          </button>
        )}
      </div>

      {showGlossary && (
        <input
          autoFocus
          aria-label="Expected terms, comma-separated"
          value={glossary}
          onChange={(e) => setGlossary(e.target.value)}
          placeholder="Kubernetes, RAG, Priya, CUDA"
          className="glass h-9 w-full rounded-md px-3 font-mono text-[12px] text-white placeholder:text-white/40 focus:outline-none"
        />
      )}

      {shown && (
        <div
          role="alert"
          className={cx(
            "glass flex items-center gap-3 rounded-lg border-red-300/50! bg-red-500/15! px-3 py-2.5",
            wide ? "absolute top-[-8px] left-[370px] w-[440px]" : "w-full",
          )}
        >
          <div className="flex-1">
            <div className="text-[13px] font-medium text-red-100">{shown.user_message}</div>
            <div className="font-mono text-[10px] text-white/45">{shown.code}</div>
          </div>
          <button onClick={retry} className="shrink-0 rounded border border-white/40 px-2 py-1 font-mono text-[10.5px] text-white hover:bg-white/10">
            TRY ANOTHER FILE
          </button>
        </div>
      )}
    </div>
  );

  return (
    <div className="landing-bg relative min-h-full overflow-hidden text-white">
      <TopRightBlob className="pointer-events-none absolute -top-[70px] -right-[60px] w-[clamp(220px,28vw,420px)]" />
      <BottomLeftBlob className="pointer-events-none absolute bottom-[3%] -left-[78px] w-[clamp(170px,25vw,340px)]" />

      {wide ? (
        <div
          className="absolute top-0 left-0"
          style={{ width: STAGE_W, height: STAGE_H, transform: `translate(${stage.x}px, ${stage.y}px) scale(${stage.scale})`, transformOrigin: "0 0" }}
        >
          {card}
          {actions}
          <div className="absolute top-[96px] left-[440px]">
            <HeroIllustration />
          </div>
        </div>
      ) : (
        <div className="relative flex min-h-full flex-col items-center px-4 pt-16 pb-28">
          {card}
          {actions}
        </div>
      )}

      <div className={cx("absolute z-10", wide ? "right-[2.4%] bottom-[7%]" : "right-4 bottom-5")}>
        <DatePill />
      </div>
    </div>
  );
}
