// Throwaway preview page used during 14-zone migration to verify the new
// 3x3 + 4-quadrant strike zone widget renders correctly. Activated by the
// ?preview=zone URL parameter; remove this file + the main.tsx hook when
// the visual check is done.

import { useState } from "react";
import { StrikeZone } from "./StrikeZone";

// Mock observed pitches across in-zone and OOZ to exercise all the
// rendering paths (in-zone dots in cells 0..8, OOZ dots in the 4 quadrants).
const MOCK_OBSERVED: { zone: number; type: string }[] = [
  { zone: 4, type: "FF" },   // middle in-zone (SIS 5)
  { zone: 0, type: "SL" },   // top-left in-zone (SIS 1)
  { zone: 8, type: "CH" },   // bottom-right in-zone (SIS 9)
  { zone: 9, type: "SL" },   // upper-left OOZ (SIS 11)
  { zone: 12, type: "FF" },  // lower-right OOZ (SIS 14)
  { zone: 2, type: "CU" },   // top-right in-zone (SIS 3)
];

// Mock whiff% per in-zone cell — 9 cells in v5 SIS.
const MOCK_WHIFF: (number | null)[] = [
  0.05, 0.10, 0.15,   // top row (z0..z2)
  0.20, 0.50, 0.25,   // middle row — high whiff in dead center
  0.30, null, 0.10,   // bottom row — null for cell 7 (no data)
];

function parseInitialState() {
  const p = new URLSearchParams(window.location.search);
  const s = p.get("selected");
  const stand = p.get("stand");
  const hl = p.get("hl");
  return {
    selected: s === null || s === "null" ? null : Number(s),
    batterStand: (stand === "L" ? "L" : "R") as "R" | "L",
    highlightedPitchIndex: hl === null || hl === "null" ? null : Number(hl),
  };
}

export function StrikeZonePreview() {
  const init = parseInitialState();
  const [selected, setSelected] = useState<number | null>(init.selected);
  const [highlightedPitchIndex, setHighlightedPitchIndex] = useState<number | null>(init.highlightedPitchIndex);
  const [batterStand, setBatterStand] = useState<"R" | "L">(init.batterStand);

  return (
    <div style={{ padding: 40, maxWidth: 720 }}>
      <h2 style={{ marginTop: 0, fontFamily: "Inter, sans-serif" }}>
        StrikeZone preview — 14-zone (v5 SIS) migration check
      </h2>
      <div style={{ marginBottom: 16, fontSize: 14, color: "#444" }}>
        Selected zone: <code>{selected === null ? "null (model samples)" : selected}</code>
        &nbsp;|&nbsp;
        Highlighted pitch index: <code>{highlightedPitchIndex ?? "none"}</code>
      </div>
      <div style={{ marginBottom: 16, fontSize: 13 }}>
        <button onClick={() => setBatterStand(batterStand === "R" ? "L" : "R")}>
          Toggle batter stand (current: {batterStand})
        </button>
        &nbsp;
        <button onClick={() => setHighlightedPitchIndex((i) => (i === null ? 0 : i === MOCK_OBSERVED.length - 1 ? null : i + 1))}>
          Cycle highlight ({highlightedPitchIndex ?? "none"})
        </button>
        &nbsp;
        <button onClick={() => setSelected(null)}>Reset selection</button>
      </div>

      <StrikeZone
        selected={selected}
        onSelect={setSelected}
        batterStand={batterStand}
        observedPitches={MOCK_OBSERVED}
        highlightedPitchIndex={highlightedPitchIndex}
        batterWhiffGrid={MOCK_WHIFF}
        disabled={false}
      />

      <hr style={{ margin: "32px 0" }} />
      <details>
        <summary style={{ cursor: "pointer", fontSize: 13, color: "#666" }}>
          Mock data used (click to expand)
        </summary>
        <pre style={{ fontSize: 11, background: "#f4f4f4", padding: 12, borderRadius: 6 }}>
{JSON.stringify({ MOCK_OBSERVED, MOCK_WHIFF }, null, 2)}
        </pre>
      </details>
    </div>
  );
}
