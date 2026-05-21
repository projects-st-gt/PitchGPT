"""Unit tests for the per-pitcher n-gram baseline."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from data.dataset import N_PITCH_TYPES, PITCH_TYPE_TO_ID
from eval.baselines.pitcher_ngram import (
    BEGIN,
    PitcherNgramBaseline,
    _add_prev_pitches,
)


def _build_pitches(rows):
    """rows: list of (pitcher, game_pk, at_bat, pitch_num, balls, strikes, pitch_type)."""
    return pd.DataFrame([
        {"pitcher": r[0], "game_pk": r[1], "at_bat_number": r[2],
         "pitch_number": r[3], "balls": r[4], "strikes": r[5],
         "pitch_type_canonical": r[6]}
        for r in rows
    ])


# ---------- _add_prev_pitches ----------


def test_add_prev_pitches_first_pitch_uses_BEGIN():
    df = _build_pitches([
        (1, 100, 1, 1, 0, 0, "FF"),
        (1, 100, 1, 2, 0, 1, "SL"),
    ])
    out = _add_prev_pitches(df, n=1)
    assert out.iloc[0]["prev_1"] == BEGIN
    assert out.iloc[1]["prev_1"] == "FF"


def test_add_prev_pitches_resets_per_at_bat():
    df = _build_pitches([
        (1, 100, 1, 1, 0, 0, "FF"),
        (1, 100, 1, 2, 0, 1, "SL"),
        (1, 100, 2, 1, 0, 0, "CH"),  # new AB → prev_1 should be BEGIN
    ])
    out = _add_prev_pitches(df, n=1).sort_values(
        ["game_pk", "at_bat_number", "pitch_number"]
    )
    assert out.iloc[0]["prev_1"] == BEGIN  # AB 1 pitch 1
    assert out.iloc[1]["prev_1"] == "FF"   # AB 1 pitch 2
    assert out.iloc[2]["prev_1"] == BEGIN  # AB 2 pitch 1


def test_add_prev_pitches_n_zero_is_identity():
    df = _build_pitches([(1, 100, 1, 1, 0, 0, "FF")])
    out = _add_prev_pitches(df, n=0)
    # n=0 → no prev_* columns added
    assert "prev_1" not in out.columns


# ---------- PitcherNgramBaseline ----------


def test_n0_per_pitcher_distinguishes_pitchers():
    """Two pitchers with very different mixes; n=0 should distinguish them
    because the conditioning includes pitcher identity."""
    rows = []
    # Pitcher 1: 80% FF, 20% SL across all counts
    pid_count = 0
    for ab in range(1, 21):
        rows.append((1, 100, ab, 1, 0, 0, "FF" if ab <= 16 else "SL"))
    # Pitcher 2: 10% FF, 90% CH
    for ab in range(21, 41):
        rows.append((2, 100, ab, 1, 0, 0, "FF" if ab <= 22 else "CH"))
    df = _build_pitches(rows)

    nb = PitcherNgramBaseline(n=0, alpha=0.0).fit(df)  # alpha=0 → no smoothing
    # Pitcher 1's rows are 0–19, pitcher 2's are 20–39. Pick one of each.
    pred = nb.predict(df.iloc[[0, 25]])
    # Pitcher 1's modal pitch is FF; Pitcher 2's modal pitch is CH
    assert pred[0] == PITCH_TYPE_TO_ID["FF"]
    assert pred[1] == PITCH_TYPE_TO_ID["CH"]


def test_n1_uses_prev_pitch_in_ab():
    """Pitcher who throws fastball after a fastball, slider after a slider —
    n=1 should pick this up."""
    rows = []
    # Pitcher 1: in 0-1 count, after FF throws FF; after SL throws SL
    for ab in range(1, 21):
        rows.append((1, 100, ab, 1, 0, 0, "FF"))
        rows.append((1, 100, ab, 2, 0, 1, "FF"))   # FF after FF
    for ab in range(21, 41):
        rows.append((1, 100, ab, 1, 0, 0, "SL"))
        rows.append((1, 100, ab, 2, 0, 1, "SL"))   # SL after SL
    df = _build_pitches(rows)

    nb = PitcherNgramBaseline(n=1, alpha=0.0).fit(df)

    # Predict for "0-1 count, prev was FF" → should predict FF.
    test_after_ff = _build_pitches([(1, 200, 1, 2, 0, 1, "?")])
    test_after_ff = pd.concat([
        _build_pitches([(1, 200, 1, 1, 0, 0, "FF")]), test_after_ff
    ], ignore_index=True)
    preds = nb.predict(test_after_ff)
    # The second row is "0-1 count after FF" → FF
    assert preds[1] == PITCH_TYPE_TO_ID["FF"]

    # Predict for "0-1 count after SL" → should predict SL
    test_after_sl = pd.concat([
        _build_pitches([(1, 201, 1, 1, 0, 0, "SL")]),
        _build_pitches([(1, 201, 1, 2, 0, 1, "?")]),
    ], ignore_index=True)
    preds = nb.predict(test_after_sl)
    assert preds[1] == PITCH_TYPE_TO_ID["SL"]


def test_smoothing_alpha_pulls_toward_league():
    """High alpha should make the per-pitcher posterior look more like
    the league prior."""
    # Build training data where league mostly throws FF, but pitcher 1
    # only throws SL (small sample).
    rows = []
    # League: 100 pitchers, each throws 100 FFs in 0-0 counts
    pid = 1
    for p in range(2, 102):
        for ab in range(1, 101):
            rows.append((p, 100 + p, ab, 1, 0, 0, "FF"))
    # Pitcher 1: only 5 SLs in 0-0
    for ab in range(1, 6):
        rows.append((1, 1000, ab, 1, 0, 0, "SL"))
    df = _build_pitches(rows)

    # alpha=0 → no smoothing → pitcher 1 looks 100% SL
    nb_no = PitcherNgramBaseline(n=0, alpha=0.0).fit(df)
    p_no = nb_no.predict_proba(_build_pitches([(1, 9999, 1, 1, 0, 0, "?")]))[0]
    assert math.isclose(float(p_no[PITCH_TYPE_TO_ID["SL"]]), 1.0)

    # alpha=1000 (>> 5 sample size) → posterior dominated by league FF
    nb_high = PitcherNgramBaseline(n=0, alpha=1000.0).fit(df)
    p_high = nb_high.predict_proba(_build_pitches([(1, 9999, 1, 1, 0, 0, "?")]))[0]
    assert p_high[PITCH_TYPE_TO_ID["FF"]] > p_high[PITCH_TYPE_TO_ID["SL"]]


def test_unseen_pitcher_falls_back_to_league_at_context():
    rows = []
    for ab in range(1, 101):
        rows.append((1, 100, ab, 1, 0, 0, "FF"))
    rows.append((1, 100, 101, 1, 3, 2, "SL"))
    df = _build_pitches(rows)

    nb = PitcherNgramBaseline(n=0, alpha=0.0).fit(df)
    # Unknown pitcher 999 in 0-0 → should fall back to 0-0 league prior (100% FF)
    out = nb.predict_proba(_build_pitches([(999, 200, 1, 1, 0, 0, "?")]))[0]
    assert math.isclose(float(out[PITCH_TYPE_TO_ID["FF"]]), 1.0)


def test_proba_sums_to_one_per_row():
    rows = []
    for ab in range(1, 11):
        rows.append((1, 100, ab, 1, 0, 0, "FF"))
        rows.append((1, 100, ab, 2, 0, 1, "SL"))
    df = _build_pitches(rows)
    nb = PitcherNgramBaseline(n=1, alpha=10.0).fit(df)
    probs = nb.predict_proba(df)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_validates_required_columns():
    with pytest.raises(KeyError, match="missing columns"):
        PitcherNgramBaseline().fit(pd.DataFrame({"pitcher": [1]}))


def test_rejects_negative_n():
    with pytest.raises(ValueError, match="n must"):
        PitcherNgramBaseline(n=-1)


def test_rejects_negative_alpha():
    with pytest.raises(ValueError, match="alpha must"):
        PitcherNgramBaseline(n=0, alpha=-0.5)
