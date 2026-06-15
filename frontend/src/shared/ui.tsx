// Shared presentational primitives used across demo tabs.
//
// Extracted from App.tsx and PitcherProfileTab.tsx so the design language
// stays consistent as more tabs land (FRONT-04 in ImprovementPlan.md). These
// components carry NO business logic — they are pure presentation.

import { PITCH_TYPE_GLYPHS, type PitchType } from "../types";

export function Section({
  children,
  className = "",
}: {
  children: React.ReactNode;
  className?: string;
}) {
  return <section className={`my-10 ${className}`}>{children}</section>;
}

export function SectionLabel({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-small font-medium text-gray-500 uppercase tracking-wide mb-3">
      {children}
    </div>
  );
}

export function Caption({ children }: { children: React.ReactNode }) {
  return <div className="text-small text-gray-500 mt-1 leading-snug">↳ {children}</div>;
}

// Bare pitch-type glyph (color + shape). Color alone fails ~8% of male users
// so the glyph always accompanies the color (accessibility).
export function PitchGlyph({ type, dim = false }: { type: PitchType; dim?: boolean }) {
  const { color, glyph } = PITCH_TYPE_GLYPHS[type];
  return (
    <span
      aria-hidden="true"
      className="inline-block text-center w-4"
      style={{ color: dim ? "#9ca3af" : color }}
    >
      {glyph}
    </span>
  );
}

// Glyph + type-code label. Tolerates unknown strings (e.g. "PAD") gracefully.
export function PitchBadge({ type }: { type: PitchType | string }) {
  const known = type in PITCH_TYPE_GLYPHS;
  if (!known) {
    return (
      <span className="inline-flex items-center gap-1.5 text-small font-medium text-gray-500">
        <span aria-hidden="true">·</span>
        <span>{type}</span>
      </span>
    );
  }
  const { color, glyph } = PITCH_TYPE_GLYPHS[type as PitchType];
  return (
    <span className="inline-flex items-center gap-1.5 text-small font-medium text-gray-900">
      <span aria-hidden="true" style={{ color }}>
        {glyph}
      </span>
      <span>{type}</span>
    </span>
  );
}
