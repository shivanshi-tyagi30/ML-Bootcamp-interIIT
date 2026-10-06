import { useEffect, useState } from "react";

/** "Tue, 6 Oct 05:22 pm" */
export function formatStamp(d: Date): string {
  const day = d.toLocaleDateString("en-GB", { weekday: "short" });
  const month = d.toLocaleDateString("en-GB", { month: "short" });
  const h = d.getHours() % 12 || 12;
  const mm = String(d.getMinutes()).padStart(2, "0");
  return `${day}, ${d.getDate()} ${month} ${String(h).padStart(2, "0")}:${mm} ${d.getHours() < 12 ? "am" : "pm"}`;
}

export function DatePill({ className }: { className?: string }) {
  const [now, setNow] = useState(() => new Date());
  useEffect(() => {
    const t = window.setInterval(() => setNow(new Date()), 15_000);
    return () => window.clearInterval(t);
  }, []);

  return (
    <time
      dateTime={now.toISOString()}
      className={`crystal inline-flex items-center gap-2.5 rounded-full px-5 py-2.5 text-[13px] font-medium text-white ${className ?? ""}`}
    >
      <svg viewBox="0 0 24 24" className="relative z-10 size-4" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" aria-hidden>
        <rect x="3.5" y="5" width="17" height="15.5" rx="2.5" />
        <path d="M3.5 10h17M8 3v4M16 3v4" />
        <rect x="7" y="13" width="3.5" height="3.5" rx="0.6" fill="currentColor" stroke="none" />
      </svg>
      <span className="relative z-10 tabular-nums [text-shadow:0_1px_6px_rgba(120,170,255,0.6)]">{formatStamp(now)}</span>
    </time>
  );
}
