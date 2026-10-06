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
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [theme, toggleTheme] = useTheme();

  // The original file plays from the browser; the backend never needs to serve it.
  const setAudio = useCallback((file: File | null) => {
    setAudioUrl((old) => {
      if (old) URL.revokeObjectURL(old);
      return file ? URL.createObjectURL(file) : null;
    });
  }, []);

  const start = async (file: File, glossary: string) => {
    setBusy(true);
    setUploadError(null);
    try {
      const jobId = await api.createJob(file, glossary);
      setAudio(file);
      setView({ name: "processing", jobId, fileName: file.name });
    } catch (e) {
      setUploadError(e instanceof ApiError ? e.jobError : makeError("E_INTERNAL"));
    } finally {
      setBusy(false);
    }
  };

  const openSample = async () => {
    const s = await api.getJob("sample");
    setAudio(null);
    setView({ name: "workspace", jobId: "sample", record: s.record });
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
        audioUrl={audioUrl}
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
      onStart={start}
      onSample={openSample}
      onClearError={() => setUploadError(null)}
    />
  );
}
