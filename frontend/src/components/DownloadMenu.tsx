import { useEffect, useRef, useState } from "react";
import type { Api } from "../lib/api";
import { downloadText, hasRecord, recordMarkdown, transcriptTxt } from "../lib/exports";
import type { ExportFormat, PartialRecord } from "../lib/types";
import { Button, Icon } from "./ui";

const OPTIONS: { fmt: ExportFormat; label: string; ext: string; needsRecord?: boolean; needsRefined?: boolean }[] = [
  { fmt: "txt_raw", label: "Raw transcript", ext: "txt" },
  { fmt: "txt_refined", label: "Refined transcript", ext: "txt", needsRefined: true },
  { fmt: "md", label: "Record · Markdown", ext: "md", needsRecord: true },
  { fmt: "docx", label: "Record · Word", ext: "docx", needsRecord: true },
  { fmt: "json", label: "Record · JSON", ext: "json" },
];

export function DownloadMenu({ api, jobId, record, beforeDownload }: {
  api: Api;
  jobId: string;
  record: PartialRecord;
  /** Waits for hand edits still being saved, so the server's file includes them. */
  beforeDownload?: () => Promise<void>;
}) {
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);
  const base = (record.meta.source_file ?? "meeting").replace(/\.[^.]+$/, "");

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent | KeyboardEvent) => {
      if (e instanceof KeyboardEvent ? e.key === "Escape" : !root.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", close);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", close);
    };
  }, [open]);

  const download = async (fmt: ExportFormat, ext: string) => {
    setOpen(false);
    await beforeDownload?.();
    const name = `${base}_${fmt}.${ext}`;
    const url = api.exportUrl(jobId, fmt);
    if (url) {
      const a = document.createElement("a");
      a.href = url;
      a.download = name;
      a.click();
      return;
    }
    // Mock mode: build the file in the browser from the same JSON.
    if (fmt === "txt_raw") downloadText(name, transcriptTxt(record.raw_transcript ?? []));
    else if (fmt === "txt_refined") downloadText(name, transcriptTxt(record.refined_transcript ?? []));
    else if (fmt === "md" && hasRecord(record)) downloadText(name, recordMarkdown(record), "text/markdown");
    else if (fmt === "json") downloadText(name, JSON.stringify(record, null, 2), "application/json");
  };

  return (
    <div ref={root} className="relative">
      <Button onClick={() => setOpen((o) => !o)} aria-haspopup="menu" aria-expanded={open}>
        <Icon.download /> Download <Icon.chevron className="size-3.5 text-ink-3" />
      </Button>
      {open && (
        <div role="menu" className="absolute right-0 z-40 mt-1 w-56 rounded-[2px] border border-line bg-surface p-1 shadow-lg">
          {OPTIONS.map((o) => {
            const disabled =
              (o.needsRecord && !hasRecord(record)) ||
              (o.needsRefined && !record.refined_transcript) ||
              (api.mock && o.fmt === "docx");
            return (
              <button
                key={o.fmt}
                role="menuitem"
                disabled={disabled}
                onClick={() => download(o.fmt, o.ext)}
                title={api.mock && o.fmt === "docx" ? "Word export needs the backend" : undefined}
                className="flex w-full items-center justify-between rounded-[2px] px-2.5 py-1.5 text-left text-sm hover:bg-raised disabled:cursor-not-allowed disabled:opacity-40"
              >
                {o.label}
                <span className="font-mono text-[11px] text-ink-3">.{o.ext}</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
