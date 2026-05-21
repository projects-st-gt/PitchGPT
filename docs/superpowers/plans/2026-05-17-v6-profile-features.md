# v6 Profile Features Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bump the PitchGPT pitcher profile schema from v5 (118 dims) to v6 (218 dims) by adding three new feature blocks — per-(pitch type × count) usage, per-(pitch type × batter stand) usage, and per-type ball movement (pfx_x, pfx_z) — so the model can distinguish curveballs from sliders in context and reduce FF argmax over-prediction.

**Architecture:** Three new feature functions in `data/player_profiles.py`, schema bump in `data/profile_cache.py` (drop 12 redundant entropy dims, add 112 new dims), `NEEDED_COLUMNS` extension in `scripts/build_profile_cache.py`, `pitcher_profile_dim` bump in `model/config.py`. Existing 3-step NaN fallback handles sparse cells without new code. Existing standardizer auto-detects D from the rebuilt cache. No model architecture changes.

**Tech Stack:** Python 3.12, pandas, numpy, pytest, PyTorch (PitchGPT model), Modal (training runs).

**Spec:** [`docs/superpowers/specs/2026-05-17-v6-profile-features-design.md`](../specs/2026-05-17-v6-profile-features-design.md)

**Prereq note — git:** This working tree is not currently under git (`git status` reports "not a git repository"). Each "checkpoint" step below shows the commit message text that would be used; if you want commits, run `git init && git add . && git commit -m 'pre-v6 state'` first. Otherwise treat checkpoints as logical pause-points.

---

## File Structure

### Modified files

| File | Change |
|---|---|
| `data/player_profiles.py` | Add 3 new functions (`pitcher_arsenal_by_count`, `pitcher_arsenal_by_stand`, `pitcher_movement_by_type`). Update module docstring. No removals. |
| `data/profile_cache.py` | Drop entropy import. Bump `PROFILE_SCHEMA_VERSION` 5 → 6. Update `PITCHER_FEATURE_NAMES` (remove 12 entropy slots, add 112 new slots). Replace entropy-population block in `build_pitcher_profile_vector` with 3 new population blocks. |
| `scripts/build_profile_cache.py` | Add `"pfx_x"`, `"pfx_z"`, `"stand"` to `NEEDED_COLUMNS`. |
| `model/config.py` | Bump `pitcher_profile_dim` 118 → 218. Update dimension-history comment. |
| `tests/test_profile_cache.py` | Update `expected` arithmetic in vector-length test (drop entropy, add 3 new blocks). Rename `test_schema_version_is_5` → `_is_6` and update assertion + docstring. |
| `tests/test_player_profiles.py` | Add tests for 3 new functions. |

### Files unchanged (verified)

`data/profile_cache_loader.py`, `model/pitchgpt.py`, `model/embeddings.py`, `model/pitchgpt_dataset.py`, `data/dataset.py`, `scripts/train_pitchgpt.py`, `scripts/calibrate_pitchgpt.py`, `scripts/fit_profile_standardizer.py`, `data/preprocess_pitchgpt.py`.

---

## Task 0: Pre-flight column check

**Files:** none (validation only)

**Goal:** Confirm `pfx_x`, `pfx_z`, and `stand` are present in `data/raw/<year>/` parquets for years 2017–2025, and quantify per-year NaN rates. This is the spec's D.1 step. If pre-2020 has very high `pfx_x`/`pfx_z` NaN rates, the league-mean fallback handles it cleanly, but we want to know.

- [ ] **Step 1: Run the pre-flight check**

```bash
uv run python << 'PY'
import glob, pandas as pd, numpy as np
needed = ["pfx_x", "pfx_z", "stand"]
for year in range(2017, 2026):
    files = sorted(glob.glob(f"/Users/sidthakur/Projects/PitchGPT/data/raw/{year}/*.parquet"))
    if not files:
        print(f"{year}: NO PARQUETS")
        continue
    # Sample 5 evenly-spaced files
    idx = np.linspace(0, len(files)-1, min(5, len(files))).astype(int)
    sample_files = [files[i] for i in idx]
    dfs = []
    for f in sample_files:
        dfs.append(pd.read_parquet(f, columns=[c for c in needed if c in pd.read_parquet(f, columns=None).columns][:3] if False else None))
    # Simpler: read full first file just to see columns, then read needed cols from sample
    sample = pd.read_parquet(sample_files[0])
    missing = [c for c in needed if c not in sample.columns]
    if missing:
        print(f"{year}: MISSING COLUMNS {missing}  (cols available: {len(sample.columns)})")
        continue
    df = pd.concat([pd.read_parquet(f, columns=needed) for f in sample_files], ignore_index=True)
    print(f"{year}  n={len(df):>7,}  pfx_x_NaN={df['pfx_x'].isna().mean()*100:>5.2f}%  pfx_z_NaN={df['pfx_z'].isna().mean()*100:>5.2f}%  stand_NaN={df['stand'].isna().mean()*100:>5.2f}%")
PY
```

Expected output: `pfx_x` and `pfx_z` NaN rates should be <5% for all years; `stand` NaN should be ~0%. If pre-2020 pfx NaN is >20%, flag in the task notes — does not block the plan (league-mean fallback handles it) but should be documented.

- [ ] **Step 2: Record findings**

Append a comment to the design doc (`docs/superpowers/specs/2026-05-17-v6-profile-features-design.md`) under R2 with the per-year NaN rates so future readers know the baseline.

- [ ] **Step 3: Checkpoint**

```
docs(v6): record per-year pfx NaN baseline from pre-flight check
```

---

## Task 1: Add `pitcher_arsenal_by_count` (TDD)

**Files:**
- Modify: `data/player_profiles.py`
- Modify: `tests/test_player_profiles.py`

**Goal:** New feature function returning per-(balls, strikes, pitch_type) usage fraction. Cells with zero observations produce no entries (slot stays NaN → league-mean fallback). Cells with observations but missing a type get 0.0.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_player_profiles.py`. Find the section just before the line `# pitcher_arsenal` (around line 109) and add this block after the existing `pitcher_arsenal` tests (after `test_pitcher_arsenal_takes_only_last_window_pitches`, around line 150):

```python
# -----------------------------------------------------------------
# pitcher_arsenal_by_count (v6 / 2026-05-17)
# -----------------------------------------------------------------


def test_pitcher_arsenal_by_count_returns_zero_for_unthrown_types_in_observed_cells():
    """A cell with at least one pitch produces entries for ALL 7 types (0.0 for unthrown)."""
    pitches = pd.DataFrame({
        "pitch_type_canonical": ["FF", "FF", "SL", "SL"],
        "balls": [0, 0, 0, 0],
        "strikes": [0, 0, 2, 2],
    })
    out = pitcher_arsenal_by_count(pitches)
    # (0, 0) was observed: 2 FF, 0 SL. (0, 2) was observed: 0 FF, 2 SL.
    assert out[(0, 0, "FF")] == 1.0
    assert out[(0, 0, "SL")] == 0.0
    assert out[(0, 2, "FF")] == 0.0
    assert out[(0, 2, "SL")] == 1.0


def test_pitcher_arsenal_by_count_omits_unobserved_cells():
    """A (b, s) cell with zero observations produces no entries at all."""
    pitches = pd.DataFrame({
        "pitch_type_canonical": ["FF"],
        "balls": [0],
        "strikes": [0],
    })
    out = pitcher_arsenal_by_count(pitches)
    assert (0, 0, "FF") in out
    assert (3, 0, "FF") not in out  # No 3-0 observations
    assert (1, 1, "FF") not in out


def test_pitcher_arsenal_by_count_empty_input_returns_empty():
    out = pitcher_arsenal_by_count(pd.DataFrame(columns=["pitch_type_canonical", "balls", "strikes"]))
    assert out == {}


def test_pitcher_arsenal_by_count_respects_trailing_window():
    """Only the last window_pitches matter."""
    pitches = pd.DataFrame({
        "pitch_type_canonical": ["FF"] * 5 + ["SL"] * 3,
        "balls": [0] * 8,
        "strikes": [0] * 8,
    })
    out = pitcher_arsenal_by_count(pitches, window_pitches=3)
    # Last 3 pitches were all SL on 0-0
    assert out[(0, 0, "SL")] == 1.0
    assert out[(0, 0, "FF")] == 0.0


def test_pitcher_arsenal_by_count_raises_on_missing_columns():
    pitches = pd.DataFrame({"pitch_type_canonical": ["FF"], "balls": [0]})  # no strikes
    with pytest.raises(KeyError, match="strikes"):
        pitcher_arsenal_by_count(pitches)
```

Also add `pitcher_arsenal_by_count` to the import block at the top of the test file (around line 15):

```python
from data.player_profiles import (
    before_asof,
    pitcher_arm_slot_by_type,
    pitcher_arsenal,
    pitcher_arsenal_by_count,  # v6
    # ... existing imports
)
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run pytest tests/test_player_profiles.py -k pitcher_arsenal_by_count -v 2>&1 | tail -15
```

Expected: `ImportError` on `pitcher_arsenal_by_count`. That's the right kind of failure — we haven't implemented yet.

- [ ] **Step 3: Implement the function**

Add to `data/player_profiles.py`. Place it directly after `pitcher_arsenal` (which ends around line 160), before `pitcher_last_n_starts_xwoba`:

```python
def pitcher_arsenal_by_count(
    pitcher_pitches: pd.DataFrame,
    *,
    window_pitches: int = 1000,
) -> dict[tuple[int, int, str], float]:
    """Per-(balls, strikes, pitch_type) usage fraction for a pitcher (v6).

    For each (balls, strikes) cell with at least one observation in the trailing
    window, return the fraction of pitches with each canonical pitch type. The
    7 fractions within a cell sum to 1.0.

    Cells with zero observations produce no entries — the caller writes NaN into
    the cache slot, and the loader's 3-step league-mean fallback fires.

    Cells with observations but where a type was not thrown get 0.0 for that
    type (the pitcher *can* throw it, just didn't in this count).

    Args:
        pitcher_pitches: this pitcher's pitches *already filtered* via
            ``before_asof`` and sorted chronologically. Required columns:
            ``pitch_type_canonical``, ``balls``, ``strikes``.
        window_pitches: how many trailing pitches to use.

    Returns:
        ``{(balls, strikes, pitch_type): fraction}`` for cells with at least
        one observation. Each (b, s) appears with all 7 pitch types or none.
    """
    required = {"pitch_type_canonical", "balls", "strikes"}
    missing = required - set(pitcher_pitches.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")
    if len(pitcher_pitches) == 0:
        return {}
    window = pitcher_pitches.tail(window_pitches)
    from data.dataset import PITCH_TYPES
    out: dict[tuple[int, int, str], float] = {}
    grouped = window.groupby(["balls", "strikes"], observed=True)
    for (b, s), group in grouped:
        if len(group) == 0:
            continue
        counts = group["pitch_type_canonical"].value_counts()
        total = float(counts.sum())
        for pt in PITCH_TYPES:
            out[(int(b), int(s), pt)] = float(counts.get(pt, 0)) / total
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run pytest tests/test_player_profiles.py -k pitcher_arsenal_by_count -v 2>&1 | tail -15
```

Expected: 5 passed.

- [ ] **Step 5: Checkpoint**

```
feat(profiles): add pitcher_arsenal_by_count for v6 schema
```

---

## Task 2: Add `pitcher_arsenal_by_stand` (TDD)

**Files:**
- Modify: `data/player_profiles.py`
- Modify: `tests/test_player_profiles.py`

**Goal:** New feature function returning per-(batter stand, pitch_type) usage fraction. Same semantics as Task 1 but conditioned on batter handedness instead of count.

**Note on convention.** The spec mentioned a shared private helper between Tasks 1 and 2 to enforce a consistent numerator/denominator convention. Given the differing key tuples (`(b, s, pt)` vs `(stand, pt)`), this plan keeps the two functions structurally parallel rather than extracting a helper — the convention is `value_counts() / sum` per group, iterating over all `PITCH_TYPES`. If a third per-condition arsenal is ever added (e.g., per-TTO in v7), extract a helper then.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_player_profiles.py` after the `pitcher_arsenal_by_count` tests:

```python
# -----------------------------------------------------------------
# pitcher_arsenal_by_stand (v6 / 2026-05-17)
# -----------------------------------------------------------------


def test_pitcher_arsenal_by_stand_returns_zero_for_unthrown_types_in_observed_cells():
    """A stand with at least one pitch produces entries for ALL 7 types (0.0 for unthrown)."""
    pitches = pd.DataFrame({
        "pitch_type_canonical": ["FF", "FF", "SL"],
        "stand": ["R", "R", "L"],
    })
    out = pitcher_arsenal_by_stand(pitches)
    # R was observed: 2 FF, 0 SL. L was observed: 0 FF, 1 SL.
    assert out[("R", "FF")] == 1.0
    assert out[("R", "SL")] == 0.0
    assert out[("L", "FF")] == 0.0
    assert out[("L", "SL")] == 1.0


def test_pitcher_arsenal_by_stand_omits_unobserved_stands():
    """If a pitcher only faced RHB, no L entries appear."""
    pitches = pd.DataFrame({
        "pitch_type_canonical": ["FF"],
        "stand": ["R"],
    })
    out = pitcher_arsenal_by_stand(pitches)
    assert ("R", "FF") in out
    assert ("L", "FF") not in out


def test_pitcher_arsenal_by_stand_empty_input_returns_empty():
    out = pitcher_arsenal_by_stand(pd.DataFrame(columns=["pitch_type_canonical", "stand"]))
    assert out == {}


def test_pitcher_arsenal_by_stand_respects_trailing_window():
    pitches = pd.DataFrame({
        "pitch_type_canonical": ["FF"] * 5 + ["SL"] * 3,
        "stand": ["R"] * 8,
    })
    out = pitcher_arsenal_by_stand(pitches, window_pitches=3)
    assert out[("R", "SL")] == 1.0
    assert out[("R", "FF")] == 0.0


def test_pitcher_arsenal_by_stand_raises_on_missing_columns():
    pitches = pd.DataFrame({"pitch_type_canonical": ["FF"]})
    with pytest.raises(KeyError, match="stand"):
        pitcher_arsenal_by_stand(pitches)
```

Add `pitcher_arsenal_by_stand` to the import block at the top of the test file.

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run pytest tests/test_player_profiles.py -k pitcher_arsenal_by_stand -v 2>&1 | tail -15
```

Expected: ImportError on `pitcher_arsenal_by_stand`.

- [ ] **Step 3: Implement the function**

Append to `data/player_profiles.py` directly after `pitcher_arsenal_by_count`:

```python
def pitcher_arsenal_by_stand(
    pitcher_pitches: pd.DataFrame,
    *,
    window_pitches: int = 1000,
) -> dict[tuple[str, str], float]:
    """Per-(batter stand, pitch_type) usage fraction for a pitcher (v6).

    For each batter handedness ('L' or 'R') the pitcher faced in the trailing
    window, return the fraction of pitches with each canonical pitch type.
    Stands with zero observations produce no entries (cache slot stays NaN →
    league-mean fallback).

    Args:
        pitcher_pitches: this pitcher's pitches *already filtered* via
            ``before_asof``. Required columns: ``pitch_type_canonical``,
            ``stand``.
        window_pitches: how many trailing pitches to use.

    Returns:
        ``{(stand, pitch_type): fraction}`` for stands with at least one
        observation. Each stand appears with all 7 pitch types or none.
    """
    required = {"pitch_type_canonical", "stand"}
    missing = required - set(pitcher_pitches.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")
    if len(pitcher_pitches) == 0:
        return {}
    window = pitcher_pitches.tail(window_pitches)
    from data.dataset import PITCH_TYPES
    out: dict[tuple[str, str], float] = {}
    for stand, group in window.groupby("stand", observed=True):
        if len(group) == 0:
            continue
        counts = group["pitch_type_canonical"].value_counts()
        total = float(counts.sum())
        for pt in PITCH_TYPES:
            out[(str(stand), pt)] = float(counts.get(pt, 0)) / total
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run pytest tests/test_player_profiles.py -k pitcher_arsenal_by_stand -v 2>&1 | tail -15
```

Expected: 5 passed.

- [ ] **Step 5: Checkpoint**

```
feat(profiles): add pitcher_arsenal_by_stand for v6 schema
```

---

## Task 3: Add `pitcher_movement_by_type` (TDD)

**Files:**
- Modify: `data/player_profiles.py`
- Modify: `tests/test_player_profiles.py`

**Goal:** New feature function returning per-pitch-type mean horizontal/vertical break. Types with zero observations are omitted (slot stays NaN → league-mean fallback).

- [ ] **Step 1: Write the failing test**

Append to `tests/test_player_profiles.py` after the `pitcher_arsenal_by_stand` tests:

```python
# -----------------------------------------------------------------
# pitcher_movement_by_type (v6 / 2026-05-17)
# -----------------------------------------------------------------


def test_pitcher_movement_by_type_computes_per_type_means():
    pitches = pd.DataFrame({
        "pitch_type_canonical": ["FF", "FF", "SL"],
        "pfx_x": [-1.0, 1.0, 5.0],
        "pfx_z": [10.0, 12.0, 2.0],
    })
    out = pitcher_movement_by_type(pitches)
    assert out["FF"]["pfx_x"] == 0.0   # mean of -1, 1
    assert out["FF"]["pfx_z"] == 11.0  # mean of 10, 12
    assert out["SL"]["pfx_x"] == 5.0
    assert out["SL"]["pfx_z"] == 2.0


def test_pitcher_movement_by_type_omits_unthrown_types():
    pitches = pd.DataFrame({
        "pitch_type_canonical": ["FF"],
        "pfx_x": [-1.0],
        "pfx_z": [10.0],
    })
    out = pitcher_movement_by_type(pitches)
    assert "FF" in out
    assert "SL" not in out
    assert "CU" not in out


def test_pitcher_movement_by_type_drops_nan_pitches_per_type():
    """A type with all-NaN pfx values is omitted; partial NaN uses nanmean."""
    import numpy as np
    pitches = pd.DataFrame({
        "pitch_type_canonical": ["FF", "FF", "SL", "SL"],
        "pfx_x": [1.0, np.nan, np.nan, np.nan],
        "pfx_z": [10.0, np.nan, np.nan, np.nan],
    })
    out = pitcher_movement_by_type(pitches)
    assert out["FF"]["pfx_x"] == 1.0  # nanmean of [1.0, NaN] = 1.0
    assert out["FF"]["pfx_z"] == 10.0
    assert "SL" not in out  # all SL pfx values were NaN → omitted


def test_pitcher_movement_by_type_empty_input_returns_empty():
    out = pitcher_movement_by_type(pd.DataFrame(columns=["pitch_type_canonical", "pfx_x", "pfx_z"]))
    assert out == {}


def test_pitcher_movement_by_type_respects_trailing_window():
    pitches = pd.DataFrame({
        "pitch_type_canonical": ["FF"] * 5 + ["SL"] * 3,
        "pfx_x": [1.0] * 5 + [5.0] * 3,
        "pfx_z": [10.0] * 5 + [2.0] * 3,
    })
    out = pitcher_movement_by_type(pitches, window_pitches=3)
    # Last 3 pitches were all SL
    assert "SL" in out
    assert out["SL"]["pfx_x"] == 5.0
    assert "FF" not in out


def test_pitcher_movement_by_type_raises_on_missing_columns():
    pitches = pd.DataFrame({"pitch_type_canonical": ["FF"], "pfx_x": [1.0]})  # no pfx_z
    with pytest.raises(KeyError, match="pfx_z"):
        pitcher_movement_by_type(pitches)
```

Add `pitcher_movement_by_type` to the import block at the top.

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run pytest tests/test_player_profiles.py -k pitcher_movement_by_type -v 2>&1 | tail -15
```

Expected: ImportError on `pitcher_movement_by_type`.

- [ ] **Step 3: Implement the function**

Append to `data/player_profiles.py` directly after `pitcher_arsenal_by_stand`:

```python
def pitcher_movement_by_type(
    pitcher_pitches: pd.DataFrame,
    *,
    window_pitches: int = 1000,
) -> dict[str, dict[str, float]]:
    """Per-pitch-type mean horizontal/vertical break for a pitcher (v6).

    For each pitch type the pitcher threw in the trailing window with at least
    one non-NaN ``pfx_x``/``pfx_z`` value, return the (NaN-aware) mean
    horizontal and vertical break. Types with no observations or all-NaN
    movement values are omitted (cache slot stays NaN → league-mean fallback).

    Statcast publishes pfx_x (horizontal break, feet) and pfx_z (vertical
    break, feet) from 2015 onward with near-zero NaN rates post-2020. Per
    the spec, these are the key CU vs SL disambiguator at a given arm slot.

    Args:
        pitcher_pitches: this pitcher's pitches *already filtered* via
            ``before_asof``. Required columns: ``pitch_type_canonical``,
            ``pfx_x``, ``pfx_z``.
        window_pitches: how many trailing pitches to use.

    Returns:
        ``{pitch_type: {"pfx_x": mean, "pfx_z": mean}}`` for types with at
        least one non-NaN observation in the window.
    """
    required = {"pitch_type_canonical", "pfx_x", "pfx_z"}
    missing = required - set(pitcher_pitches.columns)
    if missing:
        raise KeyError(f"missing required columns: {sorted(missing)}")
    if len(pitcher_pitches) == 0:
        return {}
    window = pitcher_pitches.tail(window_pitches)
    out: dict[str, dict[str, float]] = {}
    for pt, group in window.groupby("pitch_type_canonical", observed=True):
        valid = group.dropna(subset=["pfx_x", "pfx_z"])
        if len(valid) == 0:
            continue
        out[str(pt)] = {
            "pfx_x": float(valid["pfx_x"].mean()),
            "pfx_z": float(valid["pfx_z"].mean()),
        }
    return out
```

- [ ] **Step 4: Run the test to verify it passes**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run pytest tests/test_player_profiles.py -k pitcher_movement_by_type -v 2>&1 | tail -15
```

Expected: 6 passed.

- [ ] **Step 5: Checkpoint**

```
feat(profiles): add pitcher_movement_by_type for v6 schema
```

---

## Task 4: v6 schema — drop entropy, bump version, add new slots

**Files:**
- Modify: `data/profile_cache.py`
- Modify: `tests/test_profile_cache.py`

**Goal:** Update the schema constants. After this task the profile vector LENGTH is 218 but the build path still populates only the old slots (new slots will be NaN — that's correct intermediate state; Task 5 wires the new functions in).

- [ ] **Step 1: Update the schema-version test FIRST (TDD on the constants)**

Edit `tests/test_profile_cache.py`. Find `test_pitcher_vector_length_matches_documented_size` (around line 41) and replace it with:

```python
def test_pitcher_vector_length_matches_documented_size():
    # v6 (2026-05-17): drop 12 entropy dims; add 84 per-(type x count),
    # 14 per-(type x stand), 14 movement dims. Net 118 → 218.
    expected = (
        N_PITCH_TYPES  # arsenal
        + N_PITCH_TYPES  # mean_velo
        + N_PITCH_TYPES  # mean_spin
        + N_PITCH_TYPES * N_IN_ZONE_CELLS  # heatmap (7 × 9 = 63)
        + 6  # recent_30d_xwoba, n_pitches; days_since; recent_3s_xwoba, n; profile_conf
        + 2  # long_window_span_days, long_window_pct_current_season (v2)
        + N_PITCH_TYPES  # has_pitch flags
        + N_PITCH_TYPES  # arm_slot per pitch type (v4)
        + N_PITCH_TYPES * N_COUNT_STATES  # arsenal_{pt}_b{b}s{s} per-count (v6)  84
        + N_PITCH_TYPES * 2  # arsenal_{pt}_vs{stand} per-stand (v6)  14
        + N_PITCH_TYPES  # mean_pfx_x_{pt} (v6)  7
        + N_PITCH_TYPES  # mean_pfx_z_{pt} (v6)  7
    )
    assert PITCHER_VECTOR_LEN == expected
    assert len(PITCHER_FEATURE_NAMES) == expected
    assert len(PITCHER_FEATURE_INDEX) == expected
```

Also replace `test_schema_version_is_5` (around line 72) with:

```python
def test_schema_version_is_6():
    """Reminder to bump on feature changes; currently at 6 since the v6
    profile-feature expansion (drop 12 entropy dims, add 84 per-(type×count)
    + 14 per-(type×stand) + 14 movement; pitcher profile 118 → 218)."""
    assert PROFILE_SCHEMA_VERSION == 6
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run pytest tests/test_profile_cache.py::test_pitcher_vector_length_matches_documented_size tests/test_profile_cache.py::test_schema_version_is_6 -v 2>&1 | tail -15
```

Expected: both fail. The first because `PITCHER_VECTOR_LEN` is still 118 (and `expected` now computes to 218). The second because `PROFILE_SCHEMA_VERSION` is still 5.

- [ ] **Step 3: Drop entropy from PITCHER_FEATURE_NAMES**

Edit `data/profile_cache.py`. Find `PITCHER_FEATURE_NAMES` (around line 83). Delete these two lines (the entropy block, around lines 95-96):

```python
    # 12 conditional-pitch-type entropies, one per (balls, strikes) state
    *(f"entropy_b{b}s{s}" for b, s in COUNT_STATES),
```

- [ ] **Step 4: Add the 112 new slots to PITCHER_FEATURE_NAMES**

In the same `PITCHER_FEATURE_NAMES` list, immediately after the `arm_slot_{pt}` block (the last existing v5 entry, around line 121), append:

```python
    # v6 (2026-05-17): per-(pitch type × count) usage fraction, 7 × 12 = 84
    # dims. NaN for (b, s) cells with zero observations in the trailing window
    # — league-mean fallback handles. Cells with observations but where a type
    # was not thrown get 0.0. Targets CU/FC under-recall and FF argmax over-
    # prediction by exposing the conditional structure of pitcher arsenals.
    *(
        f"arsenal_{pt}_b{b}s{s}"
        for pt, (b, s) in itertools.product(PITCH_TYPES, COUNT_STATES)
    ),
    # v6: per-(pitch type × batter stand) usage fraction, 7 × 2 = 14 dims.
    # NaN for stands the pitcher hasn't faced — league-mean fallback handles.
    *(
        f"arsenal_{pt}_vs{stand}"
        for pt, stand in itertools.product(PITCH_TYPES, ("L", "R"))
    ),
    # v6: mean pfx_x (horizontal break, feet) per pitch type. NaN for types
    # never thrown or all-NaN window — league-mean fallback handles.
    *(f"mean_pfx_x_{pt}" for pt in PITCH_TYPES),
    # v6: mean pfx_z (vertical break, feet) per pitch type. NaN handling
    # same as pfx_x.
    *(f"mean_pfx_z_{pt}" for pt in PITCH_TYPES),
```

Note: `itertools` is already imported at the top of the file (used for the existing heatmap). If not, add `import itertools` near the top.

- [ ] **Step 5: Bump PROFILE_SCHEMA_VERSION and extend the version log**

In `data/profile_cache.py`, find the version-log docstring (lines ~50-72) and append a v6 entry just before the `PROFILE_SCHEMA_VERSION: int = ...` line:

```python
# v6 (2026-05-17): per-(pitch type × count) and per-(pitch type × batter stand)
#   arsenal usage blocks (84 + 14 dims), plus per-type pfx_x/pfx_z mean (14
#   dims). Drops the 12 entropy_b{b}s{s} dims as redundant given the new
#   conditional distribution. Pitcher vector 118 → 218. Targets the CU/FC
#   under-recall and FF argmax over-prediction diagnosed on the v5 baseline
#   (see docs/superpowers/specs/2026-05-17-v6-profile-features-design.md).
#   The batter schema is unchanged at v6 (rebuild re-tags batter vectors v6
#   for consistency).
```

Change the constant:

```python
PROFILE_SCHEMA_VERSION: int = 6
```

- [ ] **Step 6: Run the schema tests to verify they pass**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run pytest tests/test_profile_cache.py -k "schema_version or pitcher_vector_length" -v 2>&1 | tail -15
```

Expected: both pass. If `PITCHER_VECTOR_LEN` is still wrong, double-check the `itertools.product` order in Step 4 — the count-major outer loop is `PITCH_TYPES`, not `COUNT_STATES`.

- [ ] **Step 7: Checkpoint**

```
feat(profiles): bump schema to v6, add 112 new feature slots, drop entropy

Drops 12 entropy_b{b}s{s} dims (redundant given the new conditional
distribution). Adds 84 per-(type × count) + 14 per-(type × stand) + 14
movement dims. PROFILE_SCHEMA_VERSION 5 → 6. Pitcher vector 118 → 218.
```

---

## Task 5: Wire new functions into `build_pitcher_profile_vector`

**Files:**
- Modify: `data/profile_cache.py`

**Goal:** Replace the entropy-population block in `build_pitcher_profile_vector` with three new population blocks that call the v6 feature functions and write into the new slots. After this task, a real pitcher's vector has correctly populated v6 features.

- [ ] **Step 1: Drop the entropy import**

In `data/profile_cache.py`, find the imports near the top (line ~30-46). Remove `pitcher_count_conditional_entropy` from the `from data.player_profiles import (...)` block.

- [ ] **Step 2: Add the three new imports**

In the same import block, add:

```python
    pitcher_arsenal_by_count,
    pitcher_arsenal_by_stand,
    pitcher_movement_by_type,
```

- [ ] **Step 3: Remove the entropy-population block from `build_pitcher_profile_vector`**

Find the entropy block in `build_pitcher_profile_vector` (around lines 257-263, immediately after the recent-form scalars and before the long-window staleness block):

```python
    entropy = pitcher_count_conditional_entropy(
        pitcher_pitches, ...
    )
    for (b, s) in COUNT_STATES:
        vec[PITCHER_FEATURE_INDEX[f"entropy_b{b}s{s}"]] = float(
            entropy.get((b, s), float("nan"))
        )
```

Delete those lines entirely.

- [ ] **Step 4: Add the per-(type × count) population block**

In `build_pitcher_profile_vector`, just before the existing `arm_slot` population block (which uses `if "arm_angle" in pitcher_pitches.columns`), add:

```python
    # v6: per-(pitch type × count) usage fraction. Cells with zero observations
    # leave the slot at NaN (vec was initialized to NaN at the top of the
    # function) — the loader's 3-step league-mean fallback handles.
    if "balls" in pitcher_pitches.columns and "strikes" in pitcher_pitches.columns:
        by_count = pitcher_arsenal_by_count(pitcher_pitches, window_pitches=window_pitches)
        for (b, s, pt), frac in by_count.items():
            vec[PITCHER_FEATURE_INDEX[f"arsenal_{pt}_b{b}s{s}"]] = float(frac)
```

- [ ] **Step 5: Add the per-(type × stand) population block**

Directly after the per-count block:

```python
    # v6: per-(pitch type × batter stand) usage fraction. Same NaN-fallback
    # discipline.
    if "stand" in pitcher_pitches.columns:
        by_stand = pitcher_arsenal_by_stand(pitcher_pitches, window_pitches=window_pitches)
        for (stand, pt), frac in by_stand.items():
            vec[PITCHER_FEATURE_INDEX[f"arsenal_{pt}_vs{stand}"]] = float(frac)
```

- [ ] **Step 6: Add the movement population block**

Directly after the per-stand block:

```python
    # v6: mean pfx_x / pfx_z per pitch type. NaN-fallback handles types not
    # thrown or all-NaN windows.
    if "pfx_x" in pitcher_pitches.columns and "pfx_z" in pitcher_pitches.columns:
        movement = pitcher_movement_by_type(pitcher_pitches, window_pitches=window_pitches)
        for pt, m in movement.items():
            vec[PITCHER_FEATURE_INDEX[f"mean_pfx_x_{pt}"]] = float(m["pfx_x"])
            vec[PITCHER_FEATURE_INDEX[f"mean_pfx_z_{pt}"]] = float(m["pfx_z"])
```

- [ ] **Step 7: Run all profile-cache tests**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run pytest tests/test_profile_cache.py tests/test_player_profiles.py -v 2>&1 | tail -25
```

Expected: all pass. The shape-check test (`assert vec.shape == (PITCHER_VECTOR_LEN,)`) auto-validates that `build_pitcher_profile_vector` returns a 218-dim vector.

- [ ] **Step 8: Checkpoint**

```
feat(profiles): wire v6 feature functions into build_pitcher_profile_vector
```

---

## Task 6: Add pfx_x, pfx_z, stand to NEEDED_COLUMNS

**Files:**
- Modify: `scripts/build_profile_cache.py`

**Goal:** Without this, the corpus loader won't pull the new columns from raw, and the v6 build functions will see KeyError.

- [ ] **Step 1: Edit `NEEDED_COLUMNS`**

In `scripts/build_profile_cache.py`, find `NEEDED_COLUMNS` (around line 49). Add three entries inside the `set([...])`:

```python
    "pfx_x",  # v6: horizontal break, feet (Statcast)
    "pfx_z",  # v6: vertical break, feet (Statcast)
    "stand",  # v6: batter handedness (L/R) for per-stand arsenal
```

- [ ] **Step 2: Smoke-check the loader**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run python -c "
from scripts.build_profile_cache import _load_corpus
from pathlib import Path
df = _load_corpus(Path('data/raw/2024'))
print(f'rows: {len(df):,}')
print('columns:', sorted(df.columns))
for c in ['pfx_x', 'pfx_z', 'stand']:
    print(f'  {c}: NaN_rate={df[c].isna().mean()*100:.2f}%')
" 2>&1 | tail -15
```

Expected: all three columns present, `pfx_x` and `pfx_z` NaN rate <5%, `stand` NaN rate ~0%.

- [ ] **Step 3: Checkpoint**

```
chore(profiles): add pfx_x, pfx_z, stand to NEEDED_COLUMNS for v6
```

---

## Task 7: Bump `pitcher_profile_dim` in model config

**Files:**
- Modify: `model/config.py`

**Goal:** Tell the model the new vector size. The Linear projection layers in `model/embeddings.py:179` and `model/pitchgpt.py:55,67,121` all read from `config.pitcher_profile_dim`, so this single change updates them all.

- [ ] **Step 1: Edit the config**

In `model/config.py` (around line 123-125), update the comment and the value:

```python
    # Pitcher profile dim (PITCHER_VECTOR_LEN from data/profile_cache.py).
    # v5 (2026-04-30) shrank 230 → 118 (14-zone migration dropped 7×16 dead
    # heatmap slots). v6 (2026-05-17) grew 118 → 218: drop 12 entropy dims,
    # add 84 per-(type × count) + 14 per-(type × stand) + 14 movement.
    pitcher_profile_dim: int = 218
```

- [ ] **Step 2: Verify the config loads and matches the cache schema**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run python -c "
from model.config import PitchGPTConfig
from data.profile_cache import PITCHER_VECTOR_LEN, PROFILE_SCHEMA_VERSION
cfg = PitchGPTConfig()
assert cfg.pitcher_profile_dim == PITCHER_VECTOR_LEN, (cfg.pitcher_profile_dim, PITCHER_VECTOR_LEN)
assert PROFILE_SCHEMA_VERSION == 6
print(f'OK: pitcher_profile_dim={cfg.pitcher_profile_dim}, schema=v{PROFILE_SCHEMA_VERSION}')
" 2>&1 | tail -5
```

Expected: `OK: pitcher_profile_dim=218, schema=v6`.

- [ ] **Step 3: Smoke-instantiate the model**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run python -c "
from model.config import PitchGPTConfig
from model.pitchgpt import PitchGPT
cfg = PitchGPTConfig()
model = PitchGPT(cfg)
print(f'OK: model built with profile_dim={cfg.pitcher_profile_dim}, params={model.num_parameters():,}')
" 2>&1 | tail -5
```

Expected: `OK: model built with profile_dim=218, params=<some number>`. If it crashes, there's a hardcoded reference somewhere we missed — search for `118` in `model/`.

- [ ] **Step 4: Checkpoint**

```
feat(model): bump pitcher_profile_dim 118 → 218 for v6 schema
```

---

## Task 8: Sanity rebuild + named-sample inspection

**Files:** none (validation step)

**Goal:** Build the v6 cache for fold 0 with `--max-games 50` (cheap), then print named-slot sample values for one starter and one reliever. Per CLAUDE.md bug-prevention discipline #1 ("Print a sample before writing the slice"), this catches feature-construction bugs before paying the full-rebuild cost.

- [ ] **Step 1: Run the sanity rebuild**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run python -m scripts.build_profile_cache --role pitcher --folds 0 --max-games 50 --out-dir data/profiles_v6_sanity 2>&1 | tail -10
```

Expected: builds without error. Output parquet at `data/profiles_v6_sanity/pitcher_fold_0.parquet`.

- [ ] **Step 2: Verify schema version and vector length**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run python -c "
import pandas as pd, numpy as np
df = pd.read_parquet('data/profiles_v6_sanity/pitcher_fold_0.parquet')
print(f'rows: {len(df):,}')
print(f'schema_version unique: {df.schema_version.unique()}')
first_vec = np.asarray(df.iloc[0].vector)
print(f'first vector length: {len(first_vec)}')
assert df.schema_version.iloc[0] == 6
assert len(first_vec) == 218
print('OK')
" 2>&1 | tail -10
```

Expected: schema_version=6, length=218.

- [ ] **Step 3: Print named samples for one starter and one reliever**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run python -c "
import pandas as pd, numpy as np
from data.profile_cache import PITCHER_FEATURE_INDEX

df = pd.read_parquet('data/profiles_v6_sanity/pitcher_fold_0.parquet')

def lookup(vec, name):
    return vec[PITCHER_FEATURE_INDEX[name]]

# Print named samples for a few pitchers (pick first 3 unique pitchers)
for pid in df['player_id'].drop_duplicates().head(3):
    sub = df[df['player_id']==pid].iloc[-1]  # latest asof
    v = np.asarray(sub['vector'])
    print(f'\\n=== pitcher_id={pid}  asof={sub[\"asof_date\"]} ===')
    print(f'  arsenal_FF (unconditional)  = {lookup(v, \"arsenal_FF\"):.3f}')
    print(f'  arsenal_FF_b0s0 (1st pitch) = {lookup(v, \"arsenal_FF_b0s0\"):.3f}')
    print(f'  arsenal_FF_b3s0 (3-0)       = {lookup(v, \"arsenal_FF_b3s0\"):.3f}   (expected: high, FF on 3-0 ~0.7-0.8)')
    print(f'  arsenal_FF_b0s2 (0-2)       = {lookup(v, \"arsenal_FF_b0s2\"):.3f}   (expected: lower than b3s0)')
    print(f'  arsenal_CU_b0s2 (0-2)       = {lookup(v, \"arsenal_CU_b0s2\"):.3f}   (expected: > arsenal_CU_b3s0)')
    print(f'  arsenal_CU_b3s0 (3-0)       = {lookup(v, \"arsenal_CU_b3s0\"):.3f}')
    print(f'  arsenal_SL_vsL              = {lookup(v, \"arsenal_SL_vsL\"):.3f}')
    print(f'  arsenal_SL_vsR              = {lookup(v, \"arsenal_SL_vsR\"):.3f}')
    print(f'  mean_pfx_x_FF (in feet)     = {lookup(v, \"mean_pfx_x_FF\"):.3f}')
    print(f'  mean_pfx_x_SL (in feet)     = {lookup(v, \"mean_pfx_x_SL\"):.3f}')
    print(f'  mean_pfx_z_FF (in feet)     = {lookup(v, \"mean_pfx_z_FF\"):.3f}   (expected: positive, ~1-1.5 for rise)')
    print(f'  mean_pfx_z_CU (in feet)     = {lookup(v, \"mean_pfx_z_CU\"):.3f}   (expected: negative, ~-0.5 to -1 for drop)')
" 2>&1 | tail -60
```

**Inspect the output by eye.** Sanity checks (these are baseball ground truths):

- For most starters, `arsenal_FF_b3s0` should be MUCH higher than `arsenal_FF_b0s2` (FF on 3-0 ≈ 0.7-0.85 league; FF on 0-2 ≈ 0.2-0.3).
- `arsenal_CU_b0s2` should generally be > `arsenal_CU_b3s0` (CU usage rises in pitcher-ahead counts).
- For RHP, `mean_pfx_x_SL` should be negative (sliders break glove-side = negative x for RHP).
- `mean_pfx_z_FF` should be positive (4-seamers rise relative to gravity).
- `mean_pfx_z_CU` should be negative or near-zero (curveballs drop).

**If any of these are systematically wrong (e.g., `arsenal_FF_b3s0 < arsenal_FF_b0s2` for most pitchers), STOP.** There's a slot-index bug. Re-check Task 4 step 4 (the `itertools.product` order — should be type-major, NOT count-major).

- [ ] **Step 4: Clean up the sanity-check artifacts**

```bash
cd /Users/sidthakur/Projects/PitchGPT && rm -rf data/profiles_v6_sanity
```

- [ ] **Step 5: Checkpoint**

```
chore(v6): pass sanity rebuild + named-sample inspection
```

---

## Task 9: End-to-end local smoke test

**Files:** none (validation step)

**Goal:** Without paying Modal cost, verify that the model can ingest the v6 cache and run a single forward pass. Catches any remaining dim-mismatch bug locally.

- [ ] **Step 1: Build a tiny v6 cache for a few days**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run python -m scripts.build_profile_cache --role both --folds 0 --max-games 20 --out-dir data/profiles_v6_smoke 2>&1 | tail -10
```

Expected: builds pitcher + batter caches for fold 0, ~20 games of asof keys.

- [ ] **Step 2: Refit standardizer on the smoke cache**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run python -c "
import sys
sys.argv = ['fit_profile_standardizer']
# Monkey-patch the cache dir to the smoke dir for this one-off
import scripts.fit_profile_standardizer as m
from pathlib import Path
m.CACHE_DIR = Path('data/profiles_v6_smoke')
m.OUT_PATH = Path('data/preprocess_artifacts/v6_smoke/profile_standardization.npz')
m.OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
m.main()
" 2>&1 | tail -10
```

Expected: prints `D=218` for the pitcher fit. Both `mean` and `std` finite ranges.

- [ ] **Step 3: Run a single forward pass through the model with v6 inputs**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run python -c "
import torch
from pathlib import Path
from data.profile_cache_loader import ProfileCache
from model.config import PitchGPTConfig
from model.pitchgpt import PitchGPT
from model.pitchgpt_dataset import (
    PitchGPTAtBatDataset, ProfileStandardizer, collate_pitchgpt_at_bats,
)
from torch.utils.data import DataLoader
import pandas as pd, glob

# Find a few real pitches that match a fold-0 asof in the smoke cache
pc_p = ProfileCache(role='pitcher', fold_id=0, profiles_dir=Path('data/profiles_v6_smoke'))
pc_b = ProfileCache(role='batter',  fold_id=0, profiles_dir=Path('data/profiles_v6_smoke'))

# Just load one augmented parquet
pitches = pd.read_parquet(sorted(glob.glob('data/augmented/2024/2024-04-*.parquet'))[0])
# We need profile_standardization.npz at the smoke path
std = ProfileStandardizer(Path('data/preprocess_artifacts/v6_smoke/profile_standardization.npz'))

ds = PitchGPTAtBatDataset(
    pitches=pitches.head(500),
    pitcher_profile_lookup=pc_p.lookup,
    batter_profile_lookup=pc_b.lookup,
    profile_standardizer=std,
)
loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0, collate_fn=collate_pitchgpt_at_bats)

cfg = PitchGPTConfig()
model = PitchGPT(cfg).eval()
batch = next(iter(loader))
print(f'pitcher_profile shape: {batch[\"pitcher_profile\"].shape}  (expected last dim=218)')
assert batch['pitcher_profile'].shape[-1] == 218
with torch.no_grad():
    out = model(
        pitcher_profile=batch['pitcher_profile'],
        batter_profile=batch['batter_profile'],
        categorical_context=batch['categorical_context'],
        pitch_factors=batch['pitch_factors'],
        intended_actions=batch['intended_actions'],
        padding_mask=batch['padding_mask'],
        arsenal=batch.get('arsenal'),
    )
print(f'forward pass OK: type logits shape = {out[\"propensity\"][\"type\"].shape}')
" 2>&1 | tail -20
```

Expected: prints `pitcher_profile shape: torch.Size([..., 218])` and `forward pass OK`. If it crashes with a shape mismatch, search the model code for hardcoded references.

- [ ] **Step 4: Clean up smoke artifacts**

```bash
cd /Users/sidthakur/Projects/PitchGPT && rm -rf data/profiles_v6_smoke data/preprocess_artifacts/v6_smoke
```

- [ ] **Step 5: Checkpoint**

```
chore(v6): pass end-to-end local smoke test (cache → standardizer → model forward)
```

---

## Task 10: Full profile-cache rebuild (all folds)

**Files:** none (long-running data step)

**Goal:** Build the production v6 caches for folds 0-4, both roles. This is the long-running step (historically several hours per fold). Run it once Tasks 0-9 have all passed.

- [ ] **Step 1: Back up the v5 cache**

```bash
cd /Users/sidthakur/Projects/PitchGPT && mv data/profiles data/profiles_v5_backup && mkdir -p data/profiles
```

This preserves v5 in case we need to revert. Restore via `rm -rf data/profiles && mv data/profiles_v5_backup data/profiles`.

- [ ] **Step 2: Kick off the full rebuild**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run python -m scripts.build_profile_cache --role both --folds 0,1,2,3,4 2>&1 | tee /tmp/v6_rebuild.log
```

This is the long step. Run in foreground or `tmux`/`screen`; do not interrupt. Expect several hours.

- [ ] **Step 3: Verify all 10 output parquets exist with schema_version=6**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run python -c "
import pandas as pd, glob
for f in sorted(glob.glob('data/profiles/*.parquet')):
    df = pd.read_parquet(f, columns=['schema_version'])
    sv = df.schema_version.iloc[0] if len(df) else 'EMPTY'
    print(f'{f}  rows={len(df):,}  schema_version={sv}')
"
```

Expected: 10 files (5 folds × 2 roles), all with `schema_version=6`. If any are still v5, that fold/role didn't rebuild and Step 2 needs to be re-run for the missing entries.

- [ ] **Step 4: Checkpoint**

```
chore(v6): rebuild profile cache for all folds, both roles
```

---

## Task 11: Refit profile standardizer

**Files:**
- Modify: `data/preprocess_artifacts/v1/profile_standardization.npz` (overwrites; old file is referenced by v5 checkpoints — back it up)

**Goal:** Compute per-dim mean and std for the v6 pitcher profile. Without this, the model loader will apply v5-shaped (118-dim) stats to v6 (218-dim) vectors and crash.

- [ ] **Step 1: Back up the v5 standardizer**

```bash
cd /Users/sidthakur/Projects/PitchGPT && cp data/preprocess_artifacts/v1/profile_standardization.npz data/preprocess_artifacts/v1/profile_standardization_v5.npz
```

This preserves the v5 file in case we need to use v5 checkpoints later.

- [ ] **Step 2: Refit**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run python -m scripts.fit_profile_standardizer 2>&1 | tail -10
```

Expected output: `pitcher: fit over <N>,XXX training-period entries (fold 0); D=218; mean∈[a, b], std∈[c, d]`. Verify `D=218` for pitcher and `D=57` for batter (batter is unchanged structurally).

- [ ] **Step 3: Sanity-check the new-block stats**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run python -c "
import numpy as np
from data.profile_cache import PITCHER_FEATURE_INDEX
z = np.load('data/preprocess_artifacts/v1/profile_standardization.npz')
mean = z['pitcher_mean']; std = z['pitcher_std']
new_slots = [n for n in PITCHER_FEATURE_INDEX if n.startswith('arsenal_') and ('_b' in n or '_vs' in n) or n.startswith('mean_pfx_')]
print(f'new-block slots: {len(new_slots)}')
for n in new_slots[:10]:
    i = PITCHER_FEATURE_INDEX[n]
    print(f'  {n}: mean={mean[i]:.3f}, std={std[i]:.3f}')
# Check for degenerate std (would crash z-scoring)
bad = [(n, std[PITCHER_FEATURE_INDEX[n]]) for n in new_slots if std[PITCHER_FEATURE_INDEX[n]] < 1e-6 or std[PITCHER_FEATURE_INDEX[n]] > 1e4]
print(f'\\ndegenerate stds: {len(bad)}')
for n, s in bad:
    print(f'  {n}: std={s}')
" 2>&1 | tail -30
```

Expected: 112 new slots reported (84 + 14 + 14). All stds finite, between 1e-6 and 1e4. If any are degenerate, there's a population bug or the league-mean fallback is producing constants for some slot.

- [ ] **Step 4: Checkpoint**

```
chore(v6): refit profile standardizer to D=218
```

---

## Task 12: Train tiny fold-0 on v6 cache (Modal)

**Files:** none (training run)

**Goal:** One Modal training run on the v6 cache, fold 0, tiny size, same hyperparams as the v5 baseline (`tiny-fold0-1778792736`): `γ=0`, `α=0`. Let the new features do the work, not the loss.

- [ ] **Step 1: Inspect the v5 baseline's training hyperparams (for parity)**

The v5 baseline checkpoint is `checkpoints_modal/tiny-fold0-1778792736`. Read its config:

```bash
head -1 /Users/sidthakur/Projects/PitchGPT/checkpoints_modal/tiny-fold0-1778792736/log.jsonl | uv run python -c "
import json, sys
d = json.loads(sys.stdin.read())
print('config:', json.dumps(d['config'], indent=2))
"
```

Verify on `type_focal_gamma`, `type_class_weight_alpha`, layer counts, learning rate, epochs. The v6 run uses the SAME values. The only thing that changes is the cache version (v6) and `pitcher_profile_dim` (218, auto-applied via `model/config.py`).

- [ ] **Step 2: Kick off the Modal training run**

The project's Modal entry point is `modal_app.py::train_remote`. The v5 baseline used:

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run modal run modal_app.py::train_remote --size tiny --fold 0 --epochs 3 --run-name tiny-fold0-v6 2>&1 | tee /tmp/v6_train.log
```

Match the v5 baseline's `--epochs` value (read from the baseline log if uncertain — likely 3). Modal writes checkpoints to its Volume at `/data/checkpoints/tiny-fold0-v6/`; pulling them locally to `checkpoints_modal/tiny-fold0-v6/` is part of the existing Modal workflow (see `modal_app.py` docstring for the pull command).

- [ ] **Step 3: Verify the run completed**

```bash
cd /Users/sidthakur/Projects/PitchGPT && tail -3 checkpoints_modal/tiny-fold0-v6/log.jsonl
```

Expected: a `final_eval` line followed by a `checkpoint_saved` line.

- [ ] **Step 4: Checkpoint**

```
chore(v6): train tiny fold-0 on v6 cache
```

---

## Task 13: Calibrate + per-class diagnostic on v6 model

**Files:** none (evaluation step)

**Goal:** Compute apples-to-apples calibration and per-class metrics on the new checkpoint, and compare against the v5 baseline using the spec's acceptance criteria.

- [ ] **Step 1: Run temperature calibration on the v6 checkpoint**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run python -m scripts.calibrate_pitchgpt --ckpt checkpoints_modal/tiny-fold0-v6/checkpoint_best.pt 2>&1 | tail -15
```

Expected output: per-head table including `type   NLL_before  ECE_before  acc  T`. Capture these.

- [ ] **Step 2: Run the per-class diagnostic on baseline AND v6**

```bash
cd /Users/sidthakur/Projects/PitchGPT && uv run python -m scripts.diagnose_type_perclass \
  --ckpt checkpoints_modal/tiny-fold0-1778792736/checkpoint_best.pt \
  --ckpt checkpoints_modal/tiny-fold0-v6/checkpoint_best.pt 2>&1 | tail -50
```

Expected output: side-by-side per-class table (true_rate, pred_rate, mean_p, precision, recall, F1, support) for both checkpoints.

- [ ] **Step 3: Compare against acceptance criteria**

Fill in the comparison table from the spec:

```
| Metric              | v5 baseline | v6 actual | target  | hard floor | pass? |
|---------------------|-------------|-----------|---------|------------|-------|
| Type ECE (before-T) | 0.0077      | ___       | ≤0.010  | ≤0.015     |       |
| Type NLL (before-T) | 1.2123      | ___       | ≤1.21   | ≤1.22      |       |
| Type top-1 accuracy | 0.4805      | ___       | ≥0.482  | ≥0.475     |       |
| CU recall           | 0.2558      | ___       | ≥0.286  | ≥0.270     |       |
| FC recall           | 0.3272      | ___       | ≥0.347  | ≥0.327     |       |
| FF over-fire (pp)   | +7.57       | ___       | ≤+6.5   | ≤+7.5      |       |
| Mean P(FF)-true(FF) | -0.0028     | ___       | ±0.01   | ±0.02      |       |
| Zone/result/AB regs |  —          | ___       | none    | <1 pp      |       |
```

**Pass criteria:**
- All "hard floor" rows must pass — no exceptions.
- The four "target" rows that motivated this work (CU recall, FC recall, FF over-fire, type ECE) should all hit the target, or at minimum hit the hard floor with CU recall improving.

If any hard floor fails, revert: `rm -rf data/profiles && mv data/profiles_v5_backup data/profiles && cp data/preprocess_artifacts/v1/profile_standardization_v5.npz data/preprocess_artifacts/v1/profile_standardization.npz`.

- [ ] **Step 4: Append the comparison table to the design doc**

Open `docs/superpowers/specs/2026-05-17-v6-profile-features-design.md` and append a "Results" section at the end with the populated table from Step 3, plus 2-3 sentences interpreting the result.

- [ ] **Step 5: Checkpoint**

```
docs(v6): record v6 vs v5 comparison results
```

---

## Notes on order and parallelism

**Sequential dependencies:**
- Tasks 1-3 (new functions) must precede Task 5 (wiring).
- Task 4 (schema) must precede Task 5 (wiring) because the FEATURE_INDEX needs the new names.
- Task 6 (NEEDED_COLUMNS) must precede Task 8 (sanity rebuild).
- Task 7 (model config) must precede Task 9 (smoke test).
- Tasks 8-9 (sanity steps) must precede Task 10 (full rebuild).
- Task 10 precedes Task 11 (standardizer needs cache).
- Task 11 precedes Task 12 (training needs standardizer).

**Independent (could be done in any order or parallel):**
- Tasks 1, 2, 3 are independent of each other.
- Tasks 6 and 7 are independent (different files).

**Long-running tasks** (block on wall-clock, no agent attention needed):
- Task 10 (full rebuild): hours
- Task 12 (Modal training): hours

A subagent-driven execution can keep the main session productive on docs/eval prep while these run.
