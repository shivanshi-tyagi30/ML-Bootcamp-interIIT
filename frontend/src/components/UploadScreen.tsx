import { useEffect, useRef, useState } from "react";
import { ACCEPTED_EXTENSIONS, MAX_UPLOAD_MB, precheckFile } from "../lib/errors";
import type { Api } from "../lib/api";
import type { JobError } from "../lib/types";
import { DatePill } from "./landing/DatePill";
import { HeroIllustration } from "./landing/HeroIllustration";
import { PixelWord } from "./landing/PixelWord";
import { RecentMeetings } from "./RecentMeetings";
import { SetupCheck } from "./SetupCheck";
import { Brand, Icon, cx } from "./ui";

interface Props {
  error: JobError | null;
  busy: boolean;
  mock: boolean;
  theme: "light" | "dark";
  onTheme: () => void;
  api: Api;
  onStart: (file: File, opts: { title: string; glossary: string }) => void;
  onOpenJob: (jobId: string) => void;
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

export function UploadScreen({ error, busy, mock, theme, onTheme, api, onStart, onOpenJob, onSample, onClearError }: Props) {
  const [file, setFile] = useState<File | null>(null);
  const [title, setTitle] = useState("");
  // Which model writes the minutes when no key is pasted: the server's own Gemini key, none, or a local model.
  const [serverLlm, setServerLlm] = useState<"cloud" | "key_required" | "local">("local");
  useEffect(() => {
    if (mock) return;
    api.health().then((h) => setServerLlm(h?.llm?.cloud ? "cloud" : h?.llm?.key_required ? "key_required" : "local")).catch(() => {});
  }, [api, mock]);
  const [apiKey, setApiKey] = useState(() => (typeof window !== "undefined" ? localStorage.getItem("trace_api_key") || "" : ""));
  const [showKeyInput, setShowKeyInput] = useState(false);
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
            Drop in any meeting recording. Trace transcribes what was said, corrects misheard technical terms, and extracts action items with exact time-linked audio proof. No made-up owners or imagined deadlines.
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
              <label htmlFor="meeting-title" className="block font-mono text-[10.5px] font-semibold tracking-[0.14em] text-ink-2">
                NAME OF THE MEETING · OPTIONAL
              </label>
              <input
                id="meeting-title"
                type="text"
                value={title}
                onChange={(e) => setTitle(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && file && !localError && !busy) {
                    onStart(file, { title: title.trim(), glossary: "" });
                  }
                }}
                placeholder="e.g. Meeting 1"
                className="mt-1.5 h-10 w-full rounded-[2px] border border-ink/30 bg-surface px-3 text-[14px] font-medium text-ink placeholder:text-ink-3 transition-colors hover:border-ink/60 focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent"
              />
            </div>

            <div className="mx-3 mb-3">
              {!showKeyInput ? (
                <button
                  type="button"
                  onClick={() => setShowKeyInput(true)}
                  className="flex w-full items-center justify-between border border-dashed border-ink/30 bg-surface/60 px-3 py-2 text-left font-mono text-[11px] text-ink-2 transition-colors hover:border-ink hover:text-ink"
                >
                  <span>{apiKey ? "API Key saved · Click to edit" : "Click to paste your API Keys"}</span>
                  {apiKey ? <span className="font-mono text-[10px] text-ink-3">saved</span> : null}
                </button>
              ) : (
                <div className="border border-ink/40 bg-surface p-2.5">
                  <div className="mb-1.5 flex items-center justify-between font-mono text-[10px] tracking-wider text-ink-2">
                    <span>GEMINI API KEY</span>
                    {apiKey && (
                      <button
                        type="button"
                        onClick={() => {
                          setApiKey("");
                          localStorage.removeItem("trace_api_key");
                          setShowKeyInput(false);
                        }}
                        className="text-[10px] text-ink-3 hover:text-bad underline"
                      >
                        remove
                      </button>
                    )}
                  </div>
                  <div className="flex gap-2">
                    <input
                      type="password"
                      value={apiKey}
                      onChange={(e) => {
                        const val = e.target.value.trim();
                        setApiKey(val);
                        if (val) {
                          localStorage.setItem("trace_api_key", val);
                        } else {
                          localStorage.removeItem("trace_api_key");
                        }
                      }}
                      placeholder="Paste API key here"
                      className="h-8 flex-1 border border-ink/30 bg-bg px-2 font-mono text-[11px] text-ink placeholder:text-ink-3 focus:border-accent focus:outline-none"
                    />
                    <button
                      type="button"
                      onClick={() => setShowKeyInput(false)}
                      className="h-8 border border-ink bg-ink px-3 font-mono text-[11px] text-surface hover:bg-accent hover:text-accent-ink"
                    >
                      Save
                    </button>
                  </div>
                </div>
              )}
              <p className="mt-1.5 font-mono text-[10px] text-ink-3">
                {apiKey
                  ? "Minutes are written by Gemini (cloud). The local model is not used."
                  : serverLlm === "cloud"
                    ? "Minutes are written by Gemini with this website's key. You can paste your own key instead."
                    : serverLlm === "key_required"
                      ? "Paste your free Gemini API key to process recordings."
                      : "No key: minutes are written by the local model on this computer (much slower)."}
              </p>
            </div>

            <div className="flex items-center justify-between gap-3 border-t border-ink px-3 py-3">
              {mock ? (
                <button onClick={onSample} className="font-mono text-[11px] tracking-[0.1em] text-ink-2 hover:text-ink">
                  OPEN SAMPLE MEETING →
                </button>
              ) : null}
              <button
                disabled={!!localError || busy}
                // With no file yet, the button opens the file picker instead.
                onClick={() => (file ? onStart(file, { title: title.trim(), glossary: "" }) : input.current?.click())}
                className="ml-auto inline-flex h-10 items-center gap-2 bg-accent px-5 text-[14px] font-bold text-accent-ink transition hover:bg-ink hover:text-accent disabled:cursor-not-allowed disabled:opacity-50"
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
                {shown.detail && <div className="mt-1 font-mono text-[11px] break-words text-ink-2">{shown.detail}</div>}
              </div>
              <button onClick={retry} className="shrink-0 border border-ink bg-surface px-2.5 py-1 text-[12px] font-medium hover:bg-ink hover:text-surface">
                Try another file
              </button>
            </div>
          )}

          {!mock && <SetupCheck api={api} />}

          <RecentMeetings api={api} onOpen={onOpenJob} />

          <div className="mt-10 flex flex-col items-end gap-5 lg:hidden">
            <Promise_ label="Owners & deadlines" value="only when stated" />
            <Promise_ label="Decisions" value="only when agreed" />
          </div>
        </section>

        <section className="relative hidden h-[560px] lg:block" aria-label="How Trace links a record to the recording">
          <div className="halftone absolute top-[40px] left-[26%] size-[430px] rounded-full" />
          <div className="absolute top-[110px] left-[-2%] origin-top-left scale-[0.98] xl:scale-[1.05]">
            <HeroIllustration />
          </div>
          <div className="absolute right-0 bottom-0 flex flex-col items-end gap-6">
            <Promise_ label="Owners & deadlines" value="only when stated" />
            <Promise_ label="Decisions" value="only when agreed" />
          </div>
        </section>
      </main>
    </div>
  );
}
