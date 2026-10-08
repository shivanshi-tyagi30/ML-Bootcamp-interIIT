import { useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

export function cx(...c: (string | false | null | undefined)[]) {
  return c.filter(Boolean).join(" ");
}

type ButtonProps = React.ButtonHTMLAttributes<HTMLButtonElement> & {
  variant?: "primary" | "secondary" | "ghost";
  size?: "sm" | "md";
};

export function Button({ variant = "secondary", size = "md", className, ...rest }: ButtonProps) {
  return (
    <button
      {...rest}
      className={cx(
        "inline-flex items-center justify-center gap-1.5 font-medium transition-colors disabled:cursor-not-allowed disabled:opacity-50",
        size === "sm" ? "h-7 px-2.5 text-xs" : "h-9 px-3.5 text-sm",
        variant === "primary" && "bg-accent font-bold text-accent-ink hover:bg-ink hover:text-accent",
        variant === "secondary" && "border border-ink/20 bg-surface text-ink hover:border-ink",
        variant === "ghost" && "text-ink-2 hover:bg-raised hover:text-ink",
        className,
      )}
    />
  );
}

export function Tabs<T extends string>({
  value,
  onChange,
  items,
  label,
}: {
  value: T;
  onChange: (v: T) => void;
  items: { value: T; label: ReactNode }[];
  label: string;
}) {
  return (
    <div role="tablist" aria-label={label} className="flex flex-wrap border border-ink/15">
      {items.map((it) => (
        <button
          key={it.value}
          role="tab"
          aria-selected={value === it.value}
          onClick={() => onChange(it.value)}
          className={cx(
            "px-2.5 py-1 font-mono text-[10.5px] tracking-[0.06em] uppercase transition-colors",
            value === it.value ? "bg-ink text-surface" : "text-ink-2 hover:bg-raised hover:text-ink",
          )}
        >
          {it.label}
        </button>
      ))}
    </div>
  );
}

/** Hover/focus tooltip rendered in a portal so scroll containers don't clip it. */
export function Tip({
  content,
  children,
  className,
  focusable = true,
}: {
  content: ReactNode;
  children: ReactNode;
  className?: string;
  /** false when the child is itself focusable (e.g. a button). */
  focusable?: boolean;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const tipRef = useRef<HTMLDivElement>(null);
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ x: number; y: number; below: boolean }>({ x: 0, y: 0, below: false });

  useLayoutEffect(() => {
    if (!open || !ref.current) return;
    const r = ref.current.getBoundingClientRect();
    const below = r.top < 90;
    const w = tipRef.current?.offsetWidth ?? 240;
    const x = Math.min(Math.max(8, r.left + r.width / 2 - w / 2), window.innerWidth - w - 8);
    setPos({ x, y: below ? r.bottom + 6 : r.top - 6, below });
  }, [open]);

  return (
    <span
      ref={ref}
      tabIndex={focusable ? 0 : undefined}
      className={className}
      onMouseEnter={() => setOpen(true)}
      onMouseLeave={() => setOpen(false)}
      onFocus={() => setOpen(true)}
      onBlur={() => setOpen(false)}
    >
      {children}
      {open &&
        createPortal(
          <div
            ref={tipRef}
            role="tooltip"
            style={{ left: pos.x, top: pos.y, transform: pos.below ? undefined : "translateY(-100%)" }}
            className="pointer-events-none fixed z-50 max-w-xs border border-ink/20 bg-surface px-2.5 py-1.5 text-xs leading-relaxed text-ink shadow-lg"
          >
            {content}
          </div>,
          document.body,
        )}
    </span>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="border border-dashed border-ink/20 px-4 py-8 text-center text-sm text-ink-3">{children}</p>;
}

/** Wordmark: orange block with a cursor-style T, then the name. */
export function Brand({ compact: _ }: { compact?: boolean } = {}) {
  return (
    <div className="flex items-center gap-2.5">
      <span className="relative grid size-8 place-items-center bg-accent text-[17px] font-bold text-accent-ink">
        T
        <span className="absolute -top-1 -left-px h-[calc(100%+6px)] w-[2px] bg-accent-deep" />
      </span>
      <span className="text-[19px] leading-none font-bold tracking-[-0.03em]">
        Trace
      </span>
    </div>
  );
}

// Small inline icons (no icon dependency).
const I = ({ d, className }: { d: string; className?: string }) => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" className={cx("size-4 shrink-0", className)} aria-hidden>
    <path d={d} />
  </svg>
);
export const Icon = {
  play: (p: { className?: string }) => <I {...p} d="M7 4.5v15l12-7.5z" />,
  pause: (p: { className?: string }) => <I {...p} d="M8 5v14M16 5v14" />,
  download: (p: { className?: string }) => <I {...p} d="M12 4v11m0 0-4-4m4 4 4-4M5 20h14" />,
  upload: (p: { className?: string }) => <I {...p} d="M12 20V9m0 0-4 4m4-4 4 4M5 4h14" />,
  sun: (p: { className?: string }) => <I {...p} d="M12 3v2m0 14v2M3 12h2m14 0h2M5.6 5.6 7 7m10 10 1.4 1.4M5.6 18.4 7 17M17 7l1.4-1.4M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8z" />,
  moon: (p: { className?: string }) => <I {...p} d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z" />,
  alert: (p: { className?: string }) => <I {...p} d="M12 8v5m0 3.5v.01M10.3 3.9 2.4 18a2 2 0 0 0 1.7 3h15.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />,
  check: (p: { className?: string }) => <I {...p} d="m5 12.5 4.5 4.5L19 7" />,
  x: (p: { className?: string }) => <I {...p} d="M6 6l12 12M18 6 6 18" />,
  file: (p: { className?: string }) => <I {...p} d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8zM14 3v5h5" />,
  chevron: (p: { className?: string }) => <I {...p} d="m6 9 6 6 6-6" />,
  plus: (p: { className?: string }) => <I {...p} d="M12 5v14M5 12h14" />,
  info: (p: { className?: string }) => <I {...p} d="M12 11v5m0-8.5v.01M12 21a9 9 0 1 1 0-18 9 9 0 0 1 0 18z" />,
};
