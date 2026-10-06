import { useRef, useState } from "react";
import { ACCEPTED_EXTENSIONS, MAX_UPLOAD_MB, precheckFile } from "../lib/errors";
import type { JobError } from "../lib/types";
import { DatePill } from "./landing/DatePill";
import { HeroIllustration } from "./landing/HeroIllustration";
import { PixelWord } from "./landing/PixelWord";
import { Brand, Icon, cx } from "./ui";

interface Props {
  error: JobError | null;
  busy: boolean;
  mock: boolean;
  theme: "light" | "dark";
  onTheme: () => void;
  onStart: (file: File, glossary: string) => void;
  onSample: () => void;
  onClearError: () => void;
}

/** "Owners → only when stated", after the PS's "Team size → 1-3 members". */
function Promise_({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col items-end gap-1.5 text-right">
      <div className="flex items-center gap-2.5 text-[22px] leading-none">
        <span className="checker" />
        <span className="font-serif italic">{label}</span>
      </div>
      <span className="hl font-serif text-[22px] leading-tight italic">{value}</span>
    </div>
  );
}

export function UploadScreen({ error, busy, mock, theme, onTheme, onStart, onSample, onClearError }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [glossary, setGlossary] = useState("");
  const [drag, setDrag] = useState(false);
  const [localError, setLocalError] = useState<JobError | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const shown = localError ?? error;

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

  return (
    <div className="relative flex min-h-full flex-col overflow-hidden bg-bg">
      <PixelWord className="pointer-events-none absolute top-[11%] -right-[10%] w-[74%] max-w-[1050px] opacity-75 max-lg:hidden" />

      <header className="relative z-10 flex items-center justify-between gap-4 px-5 py-5 sm:px-10">
        <Brand />
        <div className="flex items-center gap-2">
          <DatePill className="max-sm:px-3 max-sm:text-[12px]" />
          <button
            onClick={onTheme}
            aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} mode`}
            className="grid size-9 place-items-center border border-line bg-surface text-ink-2 hover:text-ink"
          >
            {theme === "dark" ? <Icon.sun /> : <Icon.moon />}
          </button>
        </div>
      </header>

      <main className="relative z-10 mx-auto grid w-full max-w-[1320px] flex-1 items-center gap-x-16 gap-y-12 px-5 pt-4 pb-16 sm:px-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <section>
          <div className="mb-6 flex items-center gap-2.5 font-mono text-[11px] tracking-[0.18em] text-ink-2">
            <span className="checker" /> AI MEETING ASSISTANT
          </div>

          <h1 className="leading-[0.92] tracking-[-0.045em] text-ink">
            <span className="block text-[clamp(44px,5.2vw,80px)] font-bold whitespace-nowrap">Every meeting,</span>
            <span className="-mt-1 block pl-[0.6em] font-serif text-[clamp(52px,6.3vw,98px)] font-normal tracking-[-0.02em] italic">
              on record.
            </span>
          </h1>

          <p className="mt-6 max-w-[520px] text-[16px] leading-relaxed text-ink-2">
            Drop in a recording. Trace writes down what was said, fixes the jargon it misheard, and pulls out the
            decisions and to-dos, each linked to the moment it was said. If nobody named an owner, we won't make one
            up.
          </p>

          <div className="mt-8 max-w-[560px] border border-ink bg-surface shadow-[8px_8px_0_var(--color-accent)]">
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
                "group m-3 flex cursor-pointer items-center gap-4 border border-dashed px-4 py-5 transition-colors",
                drag ? "border-accent-deep bg-accent-soft" : "border-ink/30 hover:border-ink hover:bg-raised/60",
              )}
            >
              <input
                ref={input}
                type="file"
                className="sr-only"
                accept={ACCEPTED_EXTENSIONS.map((e) => "." + e).join(",") + ",audio/*"}
                onChange={(e) => choose(e.target.files?.[0])}
              />
              <span className="grid size-11 shrink-0 place-items-center bg-ink text-surface transition-colors group-hover:bg-accent group-hover:text-accent-ink">
                {file ? <Icon.file /> : <Icon.upload />}
              </span>
              <span className="min-w-0 flex-1">
                {file ? (
                  <>
                    <span className="block truncate text-[15px] font-medium">{file.name}</span>
                    <span className="block font-mono text-[11px] text-ink-3">
                      {(file.size / 1024 / 1024).toFixed(1)} MB · click to change
                    </span>
                  </>
                ) : (
                  <>
                    <span className="block text-[15px] font-medium">
                      Drop an audio file, or <span className="underline decoration-accent decoration-2 underline-offset-4">browse</span>
                    </span>
                    <span className="block font-mono text-[11px] text-ink-3">
                      {ACCEPTED_EXTENSIONS.map((e) => e.toUpperCase()).join(" · ")} · up to {MAX_UPLOAD_MB} MB
                    </span>
                  </>
                )}
              </span>
            </label>

            <div className="mx-3 mb-3">
              <label htmlFor="glossary" className="font-mono text-[10.5px] tracking-[0.14em] text-ink-3">
                EXPECTED TERMS · OPTIONAL
              </label>
              <input
                id="glossary"
                value={glossary}
                onChange={(e) => setGlossary(e.target.value)}
                placeholder="Names and jargon, e.g. Priya, Kubernetes, RAG"
                className="mt-1 h-9 w-full border-b border-line bg-transparent text-[14px] placeholder:text-ink-3/80 focus:border-ink focus:outline-none"
              />
            </div>

            <div className="flex items-center justify-between gap-3 border-t border-ink px-3 py-3">
              {mock ? (
                <button onClick={onSample} className="font-mono text-[11px] tracking-[0.1em] text-ink-2 hover:text-ink">
                  OPEN SAMPLE MEETING →
                </button>
              ) : (
                <span className="font-mono text-[11px] text-ink-3">English speech works best</span>
              )}
              <button
                disabled={!!localError || busy}
                // With no file yet, the button opens the file picker instead.
                onClick={() => (file ? onStart(file, glossary) : input.current?.click())}
                className="inline-flex h-10 items-center gap-2 bg-accent px-5 text-[14px] font-bold text-accent-ink transition hover:bg-ink hover:text-accent disabled:cursor-not-allowed disabled:opacity-50"
              >
                {busy ? "Uploading…" : file ? "Start processing" : "Choose a recording"}
                <span aria-hidden>→</span>
              </button>
            </div>
          </div>

          {shown && (
            <div role="alert" className="mt-5 flex max-w-[560px] items-start gap-3 border border-bad bg-bad-soft p-3.5">
              <Icon.alert className="mt-0.5 text-bad" />
              <div className="flex-1">
                <div className="text-[14px] font-medium text-bad">{shown.user_message}</div>
                <div className="font-mono text-[10.5px] text-ink-3">{shown.code}</div>
              </div>
              <button onClick={retry} className="shrink-0 border border-ink bg-surface px-2.5 py-1 text-[12px] font-medium hover:bg-ink hover:text-surface">
                Try another file
              </button>
            </div>
          )}

          <div className="mt-10 flex flex-col items-end gap-5 lg:hidden">
            <Promise_ label="Owners & deadlines" value="only when stated" />
            <Promise_ label="Decisions" value="only when agreed" />
          </div>
        </section>

        <section className="relative hidden h-[560px] lg:block" aria-label="How Trace links a record to the recording">
          <div className="halftone absolute top-[40px] left-[26%] size-[430px] rounded-full" />
          <div className="absolute top-[150px] left-[-2%] origin-top-left scale-[0.98] xl:scale-[1.05]">
            <HeroIllustration />
          </div>
          <div className="absolute right-0 bottom-0 flex flex-col items-end gap-6">
            <Promise_ label="Owners & deadlines" value="only when stated" />
            <Promise_ label="Decisions" value="only when agreed" />
          </div>
        </section>
      </main>

      <footer className="relative z-10 flex flex-wrap items-center justify-between gap-2 border-t border-line px-5 py-3 font-mono text-[10.5px] tracking-[0.12em] text-ink-3 sm:px-10">
        <span>INTER IIT TECH MEET 15.0 · ML PROBLEM STATEMENT</span>
        <span>SPEECH → REFINE → RECORD</span>
      </footer>
    </div>
  );
}
