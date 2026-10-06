import { useRef, useState } from "react";
import { ACCEPTED_EXTENSIONS, MAX_UPLOAD_MB, precheckFile } from "../lib/errors";
import type { JobError } from "../lib/types";
import { Button, Icon, cx } from "./ui";

interface Props {
  error: JobError | null;
  busy: boolean;
  mock: boolean;
  onStart: (file: File, glossary: string) => void;
  onSample: () => void;
  onClearError: () => void;
}

export function UploadScreen({ error, busy, mock, onStart, onSample, onClearError }: Props) {
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

  const reset = () => {
    setFile(null);
    setLocalError(null);
    onClearError();
    if (input.current) input.current.value = "";
    input.current?.click();
  };

  return (
    <div className="mx-auto flex min-h-full max-w-2xl flex-col justify-center px-4 py-12">
      <div className="mb-8">
        <div className="mb-3 flex items-center gap-2 text-sm font-semibold tracking-wide text-accent">
          <span className="grid size-6 place-items-center rounded-md bg-accent text-[11px] font-bold text-accent-ink">T</span>
          TRACE
        </div>
        <h1 className="text-2xl font-semibold tracking-tight">Turn a meeting recording into a record you can check.</h1>
        <p className="mt-2 text-sm text-ink-2">
          You get a raw transcript, a refined transcript, and a summary with minutes, decisions and action items. Each
          item links back to the moment it was said.
        </p>
      </div>

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
          "flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-6 py-12 text-center transition-colors",
          drag ? "border-accent bg-accent-soft" : "border-line bg-surface hover:border-ink-3",
        )}
      >
        <input
          ref={input}
          type="file"
          className="sr-only"
          accept={ACCEPTED_EXTENSIONS.map((e) => "." + e).join(",") + ",audio/*"}
          onChange={(e) => choose(e.target.files?.[0])}
        />
        {file ? (
          <>
            <Icon.file className="mb-3 size-8 text-accent" />
            <div className="font-medium">{file.name}</div>
            <div className="mt-1 text-xs text-ink-3">{(file.size / 1024 / 1024).toFixed(1)} MB · click to choose another</div>
          </>
        ) : (
          <>
            <Icon.upload className="mb-3 size-8 text-ink-3" />
            <div className="font-medium">Drop an audio file here, or click to browse</div>
            <div className="mt-1 text-xs text-ink-3">
              {ACCEPTED_EXTENSIONS.map((e) => e.toUpperCase()).join(", ")} · English speech · up to {MAX_UPLOAD_MB} MB
            </div>
          </>
        )}
      </label>

      {shown && (
        <div role="alert" className="mt-4 flex items-start gap-3 rounded-lg border border-bad/30 bg-bad-soft p-4">
          <Icon.alert className="mt-0.5 text-bad" />
          <div className="flex-1">
            <div className="text-sm font-medium text-bad">{shown.user_message}</div>
            <div className="mt-0.5 font-mono text-[11px] text-ink-3">{shown.code}</div>
          </div>
          <Button size="sm" onClick={reset}>
            Try another file
          </Button>
        </div>
      )}

      <div className="mt-6">
        <label htmlFor="glossary" className="text-sm font-medium">
          Expected terms <span className="font-normal text-ink-3">(optional)</span>
        </label>
        <input
          id="glossary"
          value={glossary}
          onChange={(e) => setGlossary(e.target.value)}
          placeholder="e.g. Kubernetes, RAG, Priya, CUDA"
          className="mt-1.5 h-9 w-full rounded-md border border-line bg-surface px-3 text-sm placeholder:text-ink-3 focus:border-accent focus:outline-none"
        />
        <p className="mt-1 text-xs text-ink-3">
          Comma-separated names and technical terms. These help speech recognition spell them correctly.
        </p>
      </div>

      <div className="mt-6 flex items-center gap-3">
        <Button variant="primary" disabled={!file || !!localError || busy} onClick={() => file && onStart(file, glossary)}>
          {busy ? "Uploading…" : "Start processing"}
        </Button>
        {mock && (
          <Button variant="ghost" onClick={onSample}>
            Open sample meeting
          </Button>
        )}
      </div>

      {mock && (
        <p className="mt-6 rounded-md bg-warn-soft px-3 py-2 text-xs text-warn">
          Mock mode: no backend. Any upload replays a sample meeting. Put “nospeech” or “fail-lm2” in the file name to
          test those errors.
        </p>
      )}
    </div>
  );
}
