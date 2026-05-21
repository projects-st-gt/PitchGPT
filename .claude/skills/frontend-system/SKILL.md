---
name: frontend-system
description: Use this skill any time the project's frontend is being touched — React components, design tokens, Tailwind config, motion specs, the strike-zone visualization, the pitch timeline, the trust gauge, the refusal UX for out-of-support queries, the tipping page, or the recommender page. Trigger on mentions of UI, design, component, animation, Framer Motion, D3, Recharts, color palette, typography, Tailwind, frontend, demo, or the names of any of the listed components. The design system is strict on purpose — the demo's credibility depends on not looking like generic AI-generated UI. Read this before adding new components, modifying tokens, or shipping copy.
---

# Frontend System

The demo is the most-seen artifact in the project. It needs to look
intentional, teach epistemic humility, and resist scope creep. The design
language is "Apple-minimalist" with strict tokens and a small component
library.

## Hard rules (already in CLAUDE.md, repeated for emphasis)

- **Inter, not SF Pro.** SF Pro is licensed only for Apple platforms. Use
  Inter throughout, with `system-ui` as a fallback.
- **Color + glyph for pitch types.** Color alone fails for ~8% of male
  users. Every pitch indicator pairs a color dot with a 1-letter glyph.

## Design tokens

Defined in `frontend/src/styles/tokens.css` as CSS custom properties and
mirrored into `tailwind.config.ts` so utilities work.

**Color**

| token              | value      | usage                                  |
|--------------------|------------|----------------------------------------|
| `--bg`             | `#FAFAFA`  | page background                        |
| `--surface`        | `#FFFFFF`  | card surface                           |
| `--text`           | `#1D1D1F`  | primary text                           |
| `--text-muted`     | `#86868B`  | data labels, secondary text            |
| `--border`         | `#E5E5E5`  | hairlines (1px)                        |
| `--shadow-card`    | `rgba(0,0,0,0.04)` | the only card shadow            |
| `--accent`         | `#FF6B35`  | interactive accent (orange)            |
| `--accent-cool`    | `#4A90D9`  | secondary accent (blue)                |
| `--gauge-green`    | `#50C878`  | trust gauge green                      |
| `--gauge-yellow`   | `#E8B84A`  | trust gauge yellow                     |
| `--gauge-red`      | `#D4513D`  | trust gauge red                        |

**Pitch types** — color paired with glyph. Defined in
`frontend/src/utils/pitchEncoding.ts`:

| type | color     | glyph |
|------|-----------|-------|
| FF   | `#FF6B35` | F     |
| SI   | `#D4513D` | S     |
| FC   | `#E8956A` | C     |
| SL   | `#4A90D9` | L     |
| CU   | `#7B68EE` | U     |
| CH   | `#50C878` | H     |
| FS   | `#8B8B8B` | T     |

**Typography**

- `.text-hero` — 56px / 600 / -0.03em — page hero only
- `.text-section` — 32px / 600 / -0.02em — section titles
- `.text-body` — 17px / 400 / 1.6 — paragraphs
- `.text-data-label` — 12px / 500 / 0.05em / uppercase / `--text-muted`
- `.text-stat` — 48px / 500 / -0.02em — stat cards

**Spacing and layout**

- max content width 1200px, centered
- section padding ≥ 80px vertical
- card internal padding 24px, border-radius 12px, no border, only `--shadow-card`
- grid gutters 24px

## Motion

Framer Motion with spring physics: `stiffness: 300, damping: 30`. Two
canonical entry animations only:

```tsx
const fadeSlide = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { type: "spring", stiffness: 300, damping: 30 }
};

const fadeIn = {
  initial: { opacity: 0 },
  animate: { opacity: 1 },
  transition: { duration: 0.3 }
};
```

Pitch animations: 200ms stagger on sequential reveals. No motion for
motion's sake — every animation must communicate a state change.

Loading states: skeleton screens with subtle shimmer, never spinners.

## Component library

All under `frontend/src/components/`. Each has Storybook entries; new
components require a Storybook file before merging.

| component             | purpose                                               |
|-----------------------|-------------------------------------------------------|
| `<StrikeZone>`        | D3-rendered SVG strike zone with pitch dots + heatmap |
| `<PitchTimeline>`     | Vertical timeline of pitches in an at-bat             |
| `<PitchDot>`          | Single pitch indicator (color + glyph)                |
| `<ZoneGrid>`          | Interactive 5×5 zone grid (selection + heatmap modes) |
| `<RunValueBar>`       | Diverging horizontal bar centered at 0                |
| `<EntropyChart>`      | Recharts line chart, project-styled                   |
| `<PlayerSearch>`      | Debounced borderless search                           |
| `<StatCard>`          | Single-metric display                                 |
| `<TrustGauge>`        | The positivity gauge — see below                      |
| `<RefusalCard>`       | Out-of-support refusal with explanation               |
| `<SensitivityPanel>`  | E-value display + measured-confounder calibration     |

Component code stays free of business logic. Data shaping happens in
hooks (`frontend/src/hooks/`). State lives in Zustand (`frontend/src/stores/`).

## TrustGauge — the demo's most important component

The trust gauge surfaces the positivity check from the causal layer.
It has three states; the state is decided by π̂(a* | h):

```tsx
type TrustState = "green" | "yellow" | "red";

function trustStateFor(propensity: number): TrustState {
  if (propensity > 0.05) return "green";
  if (propensity > 0.01) return "yellow";
  return "red";
}
```

Rendering:

- **Green:** show point estimate, CI, E-value. Compact gauge in the header.
- **Yellow:** show estimate with prominent uncertainty banner
  ("This intervention is at the edge of the data's support. Treat the
  estimate as exploratory."). Wider gauge.
- **Red:** render `<RefusalCard>` *instead of* a numeric estimate.
  Copy template:
  > "We can't estimate this with the available data. {Pitcher} threw a
  > {pitch_type} in this kind of situation only {n_obs} times in the
  > training data ({propensity_pct}% of comparable pitches). Try a more
  > common alternative."

Do not let the user "force" a red estimate. The refusal is the contribution.

## Page structure

Routes (`frontend/src/pages/`):

| route                  | component                  | purpose                            |
|------------------------|----------------------------|------------------------------------|
| `/`                    | `<Landing>`                | hero + auto-playing AB             |
| `/explorer`            | `<CounterfactualExplorer>` | the "Rewrite the At-Bat" demo      |
| `/tipping`             | `<TippingDetector>`        | T_start chart + flagged starts     |
| `/recommend`           | `<PitchRecommender>`       | trust-region recommender           |
| `/about`               | `<About>`                  | architecture, identification, limits |

The `/about` page is *not optional*. It states the identification
assumptions, explains the trust gauge, lists known limitations, and links
to the ADRs and methods writeup. Do not ship without it.

## Demo cache

The Counterfactual Explorer renders pre-computed rollouts for the 8
curated at-bats. Cache is built by `make demo-cache`, output at
`inference/cache/curated.json`. Each entry contains:

- the real AB pitch sequence
- 5 alternative completions per pitch index, each with N=5000 rollout summaries
- π̂(a* | h) for each alternative
- AIPW point estimate + CI (computed during cache build, not at request time)
- E-value
- The full identification footnotes

The frontend reads from this cache for instant interaction. Free-form queries
hit the live API and may show loading skeletons.

## Copy guidelines

The demo's voice is plainspoken and confident. Avoid:

- Marketing superlatives ("revolutionary," "powerful," "unprecedented")
- Hedging that erases meaning ("might possibly perhaps")
- Causal language outside the causal-layer outputs (see `causal-layer` skill
  for the forbidden / preferred list)
- Emoji
- Sentence-case or Title Case headers? Title Case for top-level page
  titles, sentence case everywhere else, consistent across the site

Numbers in copy round to two significant figures unless precision matters.
"+0.13 runs" not "+0.1276 runs."

## Things to avoid

- **Default Tailwind aesthetics.** Generic `bg-blue-500` / `rounded-lg` /
  `shadow-md` look. Use the project tokens.
- **Material UI / Chakra / shadcn-out-of-the-box** without restyling. The
  brief is "Apple-minimalist," not "developer-default."
- **Dark mode** in v1. Adds work, dilutes the look. Defer.
- **Charts with chartjunk** — 3D, drop shadows on bars, gridlines everywhere.
  Strip to the data.
- **Spinners.** Skeletons only.
- **Forcing a numeric output when the trust gauge is red.** Refusal is the
  feature.
