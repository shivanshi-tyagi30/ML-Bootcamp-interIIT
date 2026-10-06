import { useCallback, useEffect, useState } from "react";
import { ProcessingScreen } from "./components/ProcessingScreen";
import { UploadScreen } from "./components/UploadScreen";
import { Workspace } from "./components/Workspace";
import { ApiError, api } from "./lib/api";
import { makeError } from "./lib/errors";
import type { JobError, PartialRecord } from "./lib/types";

type View =
  | { name: "upload" }
  | { name: "processing"; jobId: string; fileName: string }
  | { name: "workspace"; jobId: string; record: PartialRecord; error?: JobError };

type Theme = "light" | "dark";

function useTheme(): [Theme, () => void] {
  const [theme, setTheme] = useState<Theme>(() =>
    document.documentElement.classList.contains("dark") ? "dark" : "light",
  );
  useEffect(() => {
    document.documentElement.classList.toggle("dark", theme === "dark");
    try {
      localStorage.setItem("trace-theme", theme);
    } catch {
      /* storage unavailable */
    }
  }, [theme]);
  return [theme, () => setTheme((t) => (t === "dark" ? "light" : "dark"))];
}

export default function App() {
  const [view, setView] = useState<View>({ name: "upload" });
  const [uploadError, setUploadError] = useState<JobError | null>(null);
  const [busy, setBusy] = useState(false);
  // Audio picked in this session plays from the browser; past meetings stream from the backend.
  const [local, setLocal] = useState<{ jobId: string | null; url: string } | null>(null);
  const [theme, toggleTheme] = useTheme();

  const setAudio = useCallback((file: File | null, jobId: string | null = null) => {
    setLocal((old) => {
      if (old) URL.revokeObjectURL(old.url);
      return file ? { jobId, url: URL.createObjectURL(file) } : null;
    });
  }, []);
  const audioFor = (jobId: string) => (local?.jobId === jobId ? local.url : api.audioUrl(jobId));

  const openJob = async (jobId: string) => {
    setUploadError(null);
    try {
      const s = await api.getJob(jobId);
      const finished = s.stage === "completed" || s.stage === "failed";
      if (!finished) {
        setView({ name: "processing", jobId, fileName: s.record.meta.title ?? s.record.meta.source_file ?? "recording" });
      } else if (s.error && !s.record.raw_transcript?.length) {
        setUploadError(s.error);
      } else {
        setView({ name: "workspace", jobId, record: s.record, error: s.error ?? undefined });
      }
    } catch (e) {
      setUploadError(e instanceof ApiError ? e.jobError : makeError("E_INTERNAL"));
    }
  };

  const start = async (file: File, opts: { title: string; glossary: string }) => {
    setBusy(true);
    setUploadError(null);
    try {
      const { jobId, cached } = await api.createJob(file, opts);
      setAudio(file, jobId);
      // The same recording was processed before: open the saved result.
      if (cached) await openJob(jobId);
      else setView({ name: "processing", jobId, fileName: opts.title.trim() || file.name });
    } catch (e) {
      setUploadError(e instanceof ApiError ? e.jobError : makeError("E_INTERNAL"));
    } finally {
      setBusy(false);
    }
  };

  if (view.name === "processing")
    return (
      <ProcessingScreen
        api={api}
        jobId={view.jobId}
        fileName={view.fileName}
        onDone={(record, error) => setView({ name: "workspace", jobId: view.jobId, record, error })}
        onFail={(error) => {
          setAudio(null);
          setUploadError(error);
          setView({ name: "upload" });
        }}
      />
    );

  if (view.name === "workspace")
    return (
      <Workspace
        key={view.jobId}
        api={api}
        jobId={view.jobId}
        record={view.record}
        audioUrl={audioFor(view.jobId)}
        error={view.error}
        theme={theme}
        onTheme={toggleTheme}
        onNew={() => {
          setAudio(null);
          setView({ name: "upload" });
        }}
      />
    );

  return (
    <UploadScreen
      error={uploadError}
      busy={busy}
      mock={api.mock}
      theme={theme}
      onTheme={toggleTheme}
      api={api}
      onStart={start}
      onOpenJob={openJob}
      onSample={() => openJob("sample")}
      onClearError={() => setUploadError(null)}
    />
  );
}
