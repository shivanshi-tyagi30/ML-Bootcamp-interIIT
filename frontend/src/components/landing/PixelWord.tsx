// Faint blocky lettering behind the page, after the pixel "Tech" letters on
// the problem statement. 5 × 7 glyphs drawn as rounded cells.

const GLYPHS: Record<string, string[]> = {
  T: ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
  R: ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
  A: ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
  C: ["01111", "10000", "10000", "10000", "10000", "10000", "01111"],
  E: ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
};

export function PixelWord({ word = "TRACE", cell = 56, className }: { word?: string; cell?: number; className?: string }) {
  const letters = word.split("");
  const gap = 1;
  const width = letters.length * 6 - gap;
  return (
    <svg viewBox={`0 0 ${width * cell} ${7 * cell}`} className={className} aria-hidden>
      {letters.flatMap((ch, li) =>
        (GLYPHS[ch] ?? []).flatMap((row, y) =>
          row.split("").map((bit, x) =>
            bit === "1" ? (
              <rect
                key={`${li}-${x}-${y}`}
                x={(li * 6 + x) * cell}
                y={y * cell}
                width={cell + 0.5}
                height={cell + 0.5}
                rx={cell * 0.16}
                className="fill-peach"
              />
            ) : null,
          ),
        ),
      )}
    </svg>
  );
}
