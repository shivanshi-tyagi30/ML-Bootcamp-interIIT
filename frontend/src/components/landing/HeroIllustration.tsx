// Decorative right-hand visual on the landing screen: audio → transcript line →
// action item, the evidence chain the app is built around. 560 × 250 px.

const BARS = [
  10, 18, 26, 14, 34, 22, 44, 30, 52, 38, 26, 58, 46, 70, 54, 40, 76, 62, 88, 70, 50, 64, 40, 54, 30, 44, 24, 32, 16, 22, 12,
];

function Waveform() {
  return (
    <div className="absolute top-[127px] left-0 flex -translate-y-1/2 items-center gap-[3px]">
      {BARS.map((h, i) => {
        const centre = 1 - Math.abs(i - 18) / 18; // brightest near the playhead
        return (
          <span
            key={i}
            className="w-[3px] rounded-full"
            style={{
              height: h * 0.75,
              background: `linear-gradient(180deg, rgba(190,225,255,${0.55 + centre * 0.45}), rgba(70,140,255,${0.5 + centre * 0.4}))`,
              boxShadow: centre > 0.8 ? "0 0 8px rgba(150,210,255,0.8)" : undefined,
            }}
          />
        );
      })}
    </div>
  );
}

const Dot = ({ x, y }: { x: number; y: number }) => (
  <span
    className="absolute size-[7px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-white"
    style={{ left: x, top: y, boxShadow: "0 0 0 3px rgba(120,180,255,0.45), 0 0 12px rgba(160,210,255,0.9)" }}
  />
);

const Line = ({ w, y }: { w: number; y: number }) => (
  <span className="absolute left-[15px] h-[5px] rounded-full bg-white/25" style={{ width: w, top: y }} />
);

export function HeroIllustration() {
  return (
    <div className="relative h-[250px] w-[560px] select-none" aria-hidden>
      <Waveform />

      {/* playhead */}
      <span className="absolute top-[77px] left-[108px] h-[108px] border-l border-dotted border-white/70" />
      <span className="absolute top-[193px] left-[108px] -translate-x-1/2 rounded-full border border-white/25 bg-black/60 px-1.5 py-px font-mono text-[9.5px] text-white/90">
        02:14
      </span>

      {/* connectors */}
      <svg className="absolute inset-0 overflow-visible" width="560" height="250">
        <defs>
          <linearGradient id="hero-wire" x1="0" x2="1">
            <stop offset="0" stopColor="#cfe6ff" />
            <stop offset="1" stopColor="#5b8cff" />
          </linearGradient>
        </defs>
        <path d="M108,77 C140,77 136,105 165,105" fill="none" stroke="url(#hero-wire)" strokeWidth="1.6" />
        <path d="M354,102 C384,102 376,139 405,139" fill="none" stroke="url(#hero-wire)" strokeWidth="1.6" />
      </svg>

      {/* transcript card */}
      <div className="glass absolute top-[2px] left-[157px] h-[210px] w-[204px] rounded-[14px]">
        <div className="absolute top-[14px] left-[15px] font-mono text-[8.5px] tracking-[0.14em] text-white/60">TRANSCRIPT</div>
        <Line w={150} y={33} />
        <Line w={112} y={49} />
        <Line w={164} y={66} />
        <div className="absolute top-[81px] left-[8px] w-[188px] rounded-[6px] border border-white/60 bg-blue-500/25 px-[8px] py-[5px] shadow-[0_0_14px_rgba(120,180,255,0.35)]">
          <div className="text-[8.5px] font-medium text-sky-200">S030 · Priya</div>
          <div className="text-[9.5px] leading-tight tracking-[-0.01em] whitespace-nowrap text-white">I'll update the dashboards by Monday.</div>
        </div>
        <Line w={142} y={127} />
        <Line w={172} y={143} />
        <Line w={90} y={159} />
        <Line w={156} y={183} />
      </div>

      {/* action item cards */}
      <div className="glass absolute top-[30px] left-[393px] h-[136px] w-[166px] rounded-[12px] px-[13px] pt-[14px]">
        <div className="font-mono text-[8px] tracking-[0.14em] text-white/60">ACTION ITEM</div>
        <div className="mt-[5px] text-[12px] font-bold tracking-[-0.01em] whitespace-nowrap text-white">Update the dashboards</div>
        <dl className="mt-[9px] grid grid-cols-[50px_1fr] gap-y-[5px] text-[9px]">
          <dt className="text-white/55">Owner</dt>
          <dd className="font-semibold text-white">Priya</dd>
          <dt className="text-white/55">Due</dt>
          <dd className="font-semibold text-white">Monday</dd>
        </dl>
      </div>
      <span className="absolute top-[130px] left-[412px] flex items-center gap-1 rounded-full border border-sky-300/70 bg-blue-600/40 px-[9px] py-[2px] text-[9px] font-semibold text-white shadow-[0_0_10px_rgba(90,150,255,0.5)]">
        <svg viewBox="0 0 10 10" className="size-[7px] fill-white">
          <path d="M2 1v8l7-4z" />
        </svg>
        S030
      </span>

      <div className="glass absolute top-[180px] left-[393px] h-[62px] w-[166px] rounded-[10px] px-[14px] pt-[12px]">
        <div className="text-[11px] font-semibold text-white">Clean up the test data</div>
        <div className="mt-[7px] flex items-center gap-[10px] text-[9px]">
          <span className="text-white/55">Owner</span>
          <span className="rounded-full border border-dashed border-white/50 px-[8px] py-px text-white/80">Unspecified</span>
        </div>
      </div>

      <Dot x={108} y={77} />
      <Dot x={165} y={105} />
      <Dot x={405} y={139} />
    </div>
  );
}
