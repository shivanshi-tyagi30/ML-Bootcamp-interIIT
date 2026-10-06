import { useId } from "react";

/**
 * Glossy 3D-looking ribbons built from layered SVG strokes: a gradient body,
 * a dark underside, and a blurred highlight on top.
 */
function Ribbon({ d, width, light, className }: { d: string; width: number; light?: boolean; className?: string }) {
  const id = useId().replace(/:/g, "");
  return (
    <svg viewBox="0 0 400 300" className={className} aria-hidden overflow="visible">
      <defs>
        <linearGradient id={`b${id}`} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor={light ? "#c9d8ff" : "#8fb0ff"} />
          <stop offset="0.45" stopColor={light ? "#7d9cff" : "#4a78ff"} />
          <stop offset="1" stopColor={light ? "#3d63f0" : "#1f3fd8"} />
        </linearGradient>
        <filter id={`s${id}`} x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="7" />
        </filter>
        <filter id={`h${id}`} x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="3.5" />
        </filter>
        <filter id={`g${id}`} x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="18" />
        </filter>
      </defs>
      {/* soft glow */}
      <path d={d} fill="none" stroke="#3b6cff" strokeOpacity="0.45" strokeWidth={width * 1.1} strokeLinecap="round" filter={`url(#g${id})`} />
      {/* body */}
      <path d={d} fill="none" stroke={`url(#b${id})`} strokeWidth={width} strokeLinecap="round" />
      {/* underside shading */}
      <path d={d} fill="none" stroke="#1430b0" strokeOpacity="0.55" strokeWidth={width * 0.35} strokeLinecap="round" transform={`translate(${width * 0.12} ${width * 0.28})`} filter={`url(#s${id})`} />
      {/* top highlight */}
      <path d={d} fill="none" stroke="#ffffff" strokeOpacity="0.55" strokeWidth={width * 0.14} strokeLinecap="round" transform={`translate(${-width * 0.1} ${-width * 0.26})`} filter={`url(#h${id})`} />
      <path d={d} fill="none" stroke="#ffffff" strokeOpacity="0.35" strokeWidth={width * 0.05} strokeLinecap="round" transform={`translate(${-width * 0.06} ${-width * 0.3})`} />
    </svg>
  );
}

export function TopRightBlob({ className }: { className?: string }) {
  return <Ribbon className={className} width={78} d="M60,-60 C40,70 130,150 240,140 C330,132 380,60 360,-40" />;
}

export function BottomLeftBlob({ className }: { className?: string }) {
  return <Ribbon className={className} width={70} light d="M-60,70 C60,10 175,60 170,150 C165,240 60,290 -60,250" />;
}
