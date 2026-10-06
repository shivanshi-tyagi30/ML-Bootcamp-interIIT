export function fmtTime(s: number | undefined): string {
  if (s == null || !isFinite(s)) return "--:--";
  const t = Math.max(0, Math.floor(s));
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const sec = t % 60;
  const mm = String(m).padStart(h ? 2 : 1, "0");
  const ss = String(sec).padStart(2, "0");
  return h ? `${h}:${mm}:${ss}` : `${mm}:${ss}`;
}

export const UNSPECIFIED = "Unspecified";

export function isUnspecified(v: string | undefined | null): boolean {
  return !v || v.trim().toLowerCase() === "unspecified";
}
