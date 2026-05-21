"""Run-value tables and per-pitch run-value computation, per ADR 004.

Three pieces:

1. **RE24 by season** — for each (base state, outs) cell, the mean runs scored
   in the remainder of the half-inning. 24 cells per season. Computed
   empirically from training data only.
2. **Count-state value by season** — for each (balls, strikes) state, the mean
   run-value contribution of a pitch in that state. Empirical mean of
   ``delta_run_exp`` aggregated by count.
3. **State value lookup** — ``V(state) = RE24[base, outs] + count_value[balls,
   strikes]``. Additive decomposition (ADR 004); per-pitch run value is
   ``ΔV = V(after) − V(before)``, with terminal in-play pitches mapped via
   xwOBA-on-contact and the league wOBA-to-runs conversion.

All tables are frozen for evaluation; recompute only when the training window
changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# No magic constant: the wOBA-to-runs slope is computed empirically per
# training season by ``compute_in_play_woba_to_runs_slope`` and persisted in
# ``data/run_value/woba_to_runs.json``. Earlier drafts referenced a "Tango
# published ~0.7" value; that was a confabulation — the canonical Tango/
# FanGraphs "wOBA scale" (~1.20 for 2023) is for a different aggregation
# (runs above average per PA × wOBA-difference), not for the per-pitch
# in-play contact slope this layer uses. See the pressure-testing-claims
# skill's worked example for the verification trail.

# 8 base-state encodings as 3-bit ints. Bit 2 = on_1b, bit 1 = on_2b, bit 0 = on_3b.
BASE_STATE_LABELS: list[str] = [
    "000", "001", "010", "011", "100", "101", "110", "111",
]


def base_state_from_columns(df: pd.DataFrame) -> pd.Series:
    """Encode runner state from ``on_1b``, ``on_2b``, ``on_3b`` columns.

    Statcast stores runner ID on each base (NaN = empty). We convert to a
    3-bit integer 0–7 with on_1b in the high bit.
    """
    on1 = df["on_1b"].notna().astype(int)
    on2 = df["on_2b"].notna().astype(int)
    on3 = df["on_3b"].notna().astype(int)
    return (on1 * 4 + on2 * 2 + on3).astype("int8")


def compute_re24(df: pd.DataFrame) -> pd.DataFrame:
    """Empirical RE24 table from training data.

    Required columns:

    - ``game_pk``, ``inning``, ``inning_topbot`` — identify the half-inning
    - ``on_1b``, ``on_2b``, ``on_3b`` — runner state (NaN-or-ID encoding)
    - ``outs_when_up`` — outs at start of AB (0, 1, 2)
    - ``bat_score`` — batting team's score at start of pitch

    Returns a DataFrame with columns ``base_state`` (0–7), ``outs`` (0–2),
    ``expected_runs``, ``n``.
    """
    required = {
        "game_pk", "inning", "inning_topbot",
        "on_1b", "on_2b", "on_3b",
        "outs_when_up", "bat_score",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"compute_re24 missing columns: {sorted(missing)}")

    work = df.copy()
    work["base_state"] = base_state_from_columns(work)
    work["half_inning_id"] = (
        work["game_pk"].astype(str) + "_"
        + work["inning"].astype(str) + "_"
        + work["inning_topbot"].astype(str)
    )

    # Runs scored from this state onward = end-of-half-inning score minus the
    # batting team's score *before* the current pitch. Using ``post_bat_score``
    # for the upper bound captures runs scored on the final pitch of the
    # inning; falling back to ``bat_score`` if it's unavailable would bias
    # those rows low by the runs scored on the inning's last pitch.
    score_col = "post_bat_score" if "post_bat_score" in work.columns else "bat_score"
    end_score = work.groupby("half_inning_id")[score_col].transform("max")
    work["runs_remainder"] = end_score - work["bat_score"]

    table = (
        work.groupby(["base_state", "outs_when_up"], dropna=False)["runs_remainder"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"outs_when_up": "outs", "mean": "expected_runs", "count": "n"})
    )
    return table


def compute_count_value(df: pd.DataFrame) -> pd.DataFrame:
    """Empirical count-state run-value table.

    Implementation: mean of Statcast's per-pitch ``delta_run_exp`` grouped
    by ``(balls, strikes)``. This is the additive count component of
    ``V(state)`` per ADR 004. Refine the definition (e.g., AB-terminal-only
    estimators, regression-based versions) once we validate against held-out
    data.

    Required columns: ``balls``, ``strikes``, ``delta_run_exp``.
    """
    required = {"balls", "strikes", "delta_run_exp"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"compute_count_value missing columns: {sorted(missing)}")

    table = (
        df.groupby(["balls", "strikes"], dropna=False)["delta_run_exp"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "count_value", "count": "n"})
    )
    return table


def compute_in_play_woba_to_runs_slope(df: pd.DataFrame) -> float:
    """Empirical per-pitch slope: ``delta_run_exp ≈ k · xwOBA-on-contact``.

    Filters to in-play pitches (rows with both
    ``estimated_woba_using_speedangle`` and ``delta_run_exp`` populated),
    then OLS-fits through the origin: ``k = Σxy / Σx²``.

    What this is and isn't:

    - **Is:** the per-pitch in-play contact slope used by
      ``in_play_run_value(xwoba, slope)`` to map a hypothetical pitch's
      xwOBA-on-contact to a run-value contribution during causal rollouts.
      Empirically ~0.49 on 2023 data.
    - **Is not:** Tango/FanGraphs's "wOBA scale" (~1.20 for 2023). That
      scale is for runs-above-average per (PA × wOBA-difference) — a
      different aggregation entirely. Earlier drafts conflated the two;
      see the pressure-testing-claims skill for the verification trail.

    Required columns: ``estimated_woba_using_speedangle``, ``delta_run_exp``.
    """
    required = {"estimated_woba_using_speedangle", "delta_run_exp"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"compute_in_play_woba_to_runs_slope missing columns: {sorted(missing)}"
        )

    sub = df.dropna(subset=["estimated_woba_using_speedangle", "delta_run_exp"])
    if len(sub) == 0:
        raise ValueError("no rows with both xwOBA and delta_run_exp available")

    x = sub["estimated_woba_using_speedangle"].to_numpy(dtype=float)
    y = sub["delta_run_exp"].to_numpy(dtype=float)
    denom = np.dot(x, x)
    if denom == 0:
        raise ValueError("xwOBA values are all zero; cannot fit slope")
    return float(np.dot(x, y) / denom)


def state_value(
    re24: pd.DataFrame,
    count_value: pd.DataFrame,
    base_state: int,
    outs: int,
    balls: int,
    strikes: int,
) -> float:
    """V(state) = RE24[base, outs] + count_value[balls, strikes]."""
    re24_row = re24[(re24["base_state"] == base_state) & (re24["outs"] == outs)]
    if len(re24_row) == 0:
        raise KeyError(f"RE24 missing cell base_state={base_state}, outs={outs}")
    re_part = float(re24_row["expected_runs"].iloc[0])

    cv_row = count_value[(count_value["balls"] == balls) & (count_value["strikes"] == strikes)]
    if len(cv_row) == 0:
        raise KeyError(f"count_value missing cell balls={balls}, strikes={strikes}")
    cv_part = float(cv_row["count_value"].iloc[0])

    return re_part + cv_part


def delta_v(
    re24: pd.DataFrame,
    count_value: pd.DataFrame,
    before: tuple[int, int, int, int],
    after: tuple[int, int, int, int],
) -> float:
    """V(after) - V(before). State tuples are (base_state, outs, balls, strikes)."""
    return state_value(re24, count_value, *after) - state_value(re24, count_value, *before)


def in_play_run_value(xwoba: float, slope: float) -> float:
    """Map xwOBA-on-contact to a per-pitch run-value contribution.

    ``slope`` is the empirical per-pitch in-play contact slope from
    ``compute_in_play_woba_to_runs_slope``, persisted per training season
    in ``data/run_value/woba_to_runs.json``. Callers pass the slope for
    the relevant season — there is no default, because there is no
    season-independent canonical value (~0.49 on 2023, but this drifts
    with the run environment).
    """
    if xwoba is None or (isinstance(xwoba, float) and np.isnan(xwoba)):
        return float("nan")
    return float(xwoba) * float(slope)


def add_run_value_column(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``run_value`` column to a pitch DataFrame.

    For *actual* (observed) pitches, ``run_value`` is taken directly from
    Statcast's ``delta_run_exp`` field — MLB's authoritative per-pitch run
    value computation. This is the canonical run_value used everywhere in
    the project for observed pitches.

    For *counterfactual* (hypothetical) pitches generated during causal
    rollouts, use ``state_value`` / ``delta_v`` against our empirical
    ``re24`` and ``count_value`` tables instead — those derive V(state)
    from the frozen training-data tables.
    """
    if "delta_run_exp" not in df.columns:
        raise KeyError("add_run_value_column expects 'delta_run_exp' column from Statcast")
    df = df.copy()
    df["run_value"] = df["delta_run_exp"]
    return df
