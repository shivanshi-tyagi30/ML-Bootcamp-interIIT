// Decorative visual on the landing screen: audio → transcript line → action
// item, the evidence chain the app is built around. 560 × 260 px.

const BARS = [
  8, 14, 22, 12, 30, 20, 40, 26, 48, 34, 22, 54, 42, 66, 50, 36, 72, 58, 84, 64, 46, 60, 36, 50, 26, 40, 20, 28, 14, 18, 10,
];
const PLAYHEAD_BAR = 18;

function Waveform() {
  return (
    <div className="absolute top-[132px] left-0 flex -translate-y-1/2 items-center gap-[3px]">
      {BARS.map((h, i) => (
        <span
          key={i}
          className={i >= PLAYHEAD_BAR - 3 && i <= PLAYHEAD_BAR + 2 ? "w-[3px] bg-accent" : "w-[3px] bg-ink"}
          style={{ height: h * 0.8, opacity: i > PLAYHEAD_BAR + 2 ? 0.35 : 1 }}
        />
      ))}
    </div>
  );
}

const Node = ({ x, y }: { x: number; y: number }) => (
  <span
    className="absolute size-[9px] -translate-x-1/2 -translate-y-1/2 border-2 border-ink bg-accent"
    style={{ left: x, top: y }}
  />
);

const Line = ({ w, y }: { w: number; y: number }) => (
  <span className="absolute left-[16px] h-[4px] bg-ink/10" style={{ width: w, top: y }} />
);

export function HeroIllustration() {
  return (
    <div className="relative h-[260px] w-[560px] select-none" aria-hidden>
      <Waveform />

      {/* playhead */}
      <span className="absolute top-[78px] left-[108px] h-[112px] border-l border-dashed border-ink/60" />
      <span className="absolute top-[198px] left-[108px] -translate-x-1/2 bg-ink px-1.5 py-px font-mono text-[9.5px] text-surface">
        02:14
      </span>

      <svg className="absolute inset-0 overflow-visible" width="560" height="260">
        <path d="M108,78 C140,78 136,108 166,108" fill="none" className="stroke-ink" strokeWidth="1.2" />
        <path d="M354,108 C384,108 376,146 406,146" fill="none" className="stroke-ink" strokeWidth="1.2" />
      </svg>

      {/* transcript card */}
      <div className="absolute top-[4px] left-[158px] h-[214px] w-[204px] border border-ink/15 bg-surface shadow-[6px_6px_0_rgba(17,17,17,0.06)]">
        <div className="absolute top-[13px] left-[16px] font-mono text-[8.5px] tracking-[0.16em] text-ink-3">TRANSCRIPT</div>
        <Line w={150} y={34} />
        <Line w={112} y={50} />
        <Line w={164} y={66} />
        <div className="absolute top-[84px] left-[9px] w-[186px] px-[7px] py-[5px]">
          <div className="font-mono text-[8px] text-ink-3">S030 · PRIYA</div>
          <div className="mt-[2px] text-[10.5px] leading-[1.35] text-ink">
            I'll update the dashboards
            <br />
            <span className="hl">by Monday.</span>
          </div>
        </div>
        <Line w={142} y={134} />
        <Line w={172} y={150} />
        <Line w={90} y={166} />
        <Line w={156} y={190} />
      </div>

      {/* action items */}
      <div className="absolute top-[32px] left-[394px] h-[140px] w-[166px] border border-ink/15 bg-surface px-[14px] pt-[13px] shadow-[6px_6px_0_rgba(17,17,17,0.06)]">
        <div className="font-mono text-[8px] tracking-[0.16em] text-ink-3">ACTION ITEM</div>
        <div className="mt-[6px] text-[12.5px] leading-tight font-bold whitespace-nowrap text-ink">Update the dashboards</div>
        <dl className="mt-[10px] grid grid-cols-[46px_1fr] gap-y-[5px] text-[9.5px]">
          <dt className="text-ink-3">Owner</dt>
          <dd className="font-medium text-ink">Priya</dd>
          <dt className="text-ink-3">Due</dt>
          <dd className="font-medium text-ink">Monday</dd>
        </dl>
        <span className="absolute bottom-[14px] left-[14px] flex items-center gap-1 bg-ink px-[7px] py-[2px] font-mono text-[8.5px] text-surface">
          <svg viewBox="0 0 10 10" className="size-[6px] fill-accent">
            <path d="M2 1v8l7-4z" />
          </svg>
          S030
        </span>
      </div>

      <div className="absolute top-[186px] left-[394px] h-[62px] w-[166px] border border-ink/15 bg-surface px-[14px] pt-[11px] shadow-[6px_6px_0_rgba(17,17,17,0.06)]">
        <div className="text-[11px] font-bold text-ink">Clean up the test data</div>
        <div className="mt-[7px] flex items-center gap-[10px] text-[9px]">
          <span className="text-ink-3">Owner</span>
          <span className="rounded-full border border-dashed border-unspec/70 px-[7px] py-px text-unspec">Unspecified</span>
        </div>
      </div>

      <Node x={108} y={78} />
      <Node x={166} y={108} />
      <Node x={406} y={146} />
    </div>
  );
}
