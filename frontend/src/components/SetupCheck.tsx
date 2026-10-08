import { useEffect, useState } from "react";
import type { Api } from "../lib/api";
import type { Health } from "../lib/types";
import { Icon } from "./ui";

/** Warns before upload when the backend is missing ffmpeg, Ollama or a model. */
export function SetupCheck({ api }: { api: Api }) {
  const [h, setH] = useState<Health | null | undefined>(undefined);

  useEffect(() => {
    let alive = true;
    const load = () => api.health().then((x) => alive && setH(x));
    load();
    const t = window.setInterval(load, 15000);
    return () => {
      alive = false;
      window.clearInterval(t);
    };
  }, [api]);

  if (h === undefined) return null;
  const problems: { text: string; fix?: string }[] = [];
  if (h === null) problems.push({ text: "The backend isn't running.", fix: "uvicorn app.main:app --port 8000" });
  else {
    if (h.ffmpeg === false) problems.push({ text: "ffmpeg isn't installed, so audio can't be converted.", fix: "winget install Gyan.FFmpeg" });
    // With a cloud API key (saved in this browser or configured on the server) Ollama is not needed.
    let cloudKey = false;
    try {
      cloudKey = !!localStorage.getItem("trace_api_key")?.trim();
    } catch {
      /* storage unavailable */
    }
    if (!cloudKey && !h.llm?.cloud) {
      if (h.llm?.reachable === false) problems.push({ text: `Ollama isn't reachable at ${h.llm.host}.`, fix: "ollama serve" });
      else for (const m of h.llm?.missing ?? []) problems.push({ text: `Model ${m} isn't downloaded.`, fix: `ollama pull ${m}` });
    }
  }
  if (!problems.length) return null;

  return (
    <div role="status" className="mt-5 flex max-w-[560px] items-start gap-3 border border-warn/40 bg-warn-soft p-3.5 text-warn">
      <Icon.alert className="mt-0.5 shrink-0" />
      <div className="min-w-0 flex-1 space-y-1.5 text-[13px]">
        <div className="font-medium">Setup check: processing will fail until this is fixed</div>
        {problems.map((p) => (
          <div key={p.text} className="text-ink-2">
            {p.text}
            {p.fix && <code className="ml-1.5 bg-surface px-1.5 py-px font-mono text-[11px] text-ink">{p.fix}</code>}
          </div>
        ))}
      </div>
    </div>
  );
}
