"""Unit tests for the marginal and count-conditional baselines."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from data.dataset import PITCH_TYPE_TO_ID, N_PITCH_TYPES
from eval.baselines.count_conditional import CountConditionalBaseline
from eval.baselines.marginal import MarginalBaseline


# ---------- MarginalBaseline ----------


def test_marginal_predicts_constant_distribution():
    df = pd.DataFrame({"pitch_type_canonical": ["FF"] * 6 + ["SL"] * 4})
    m = MarginalBaseline().fit(df)
    probs = m.predict_proba(pd.DataFrame({"balls": [0, 1, 2]}))
    assert probs.shape == (3, N_PITCH_TYPES)
    # All rows are the same constant distribution
    np.testing.assert_allclose(probs[0], probs[1])
    np.testing.assert_allclose(probs[1], probs[2])
    # FF = 60%, SL = 40%
    assert math.isclose(float(probs[0, PITCH_TYPE_TO_ID["FF"]]), 0.6)
    assert math.isclose(float(probs[0, PITCH_TYPE_TO_ID["SL"]]), 0.4)


def test_marginal_predict_returns_mode():
    df = pd.DataFrame({"pitch_type_canonical": ["FF"] * 6 + ["SL"] * 4})
    m = MarginalBaseline().fit(df)
    preds = m.predict(pd.DataFrame({"balls": [0, 0, 0]}))
    assert (preds == PITCH_TYPE_TO_ID["FF"]).all()


def test_marginal_most_common_type():
    df = pd.DataFrame({"pitch_type_canonical": ["FF"] * 100 + ["SL"] * 30 + ["CH"] * 5})
    m = MarginalBaseline().fit(df)
    assert m.most_common_type == "FF"


def test_marginal_drops_unmapped_pitch_types():
    """Unknown pitch types contribute nothing to the trained distribution."""
    df = pd.DataFrame({"pitch_type_canonical": ["FF"] * 5 + ["UNKNOWN_PITCH"] * 5})
    m = MarginalBaseline().fit(df)
    # Only FF is in canonical map → 100% FF after normalization
    assert math.isclose(float(m.probs_[PITCH_TYPE_TO_ID["FF"]]), 1.0)


def test_marginal_falls_back_to_uniform_when_no_canonical_pitches():
    df = pd.DataFrame({"pitch_type_canonical": ["NOT_A_PITCH"] * 5})
    m = MarginalBaseline().fit(df)
    # Degenerate case — uniform distribution
    np.testing.assert_allclose(m.probs_, np.full(N_PITCH_TYPES, 1.0 / N_PITCH_TYPES))


def test_marginal_raises_when_predict_before_fit():
    m = MarginalBaseline()
    with pytest.raises(RuntimeError, match="fit"):
        m.predict(pd.DataFrame({"balls": [0]}))


def test_marginal_validates_required_column():
    with pytest.raises(KeyError, match="pitch_type_canonical"):
        MarginalBaseline().fit(pd.DataFrame({"foo": [1, 2]}))


# ---------- CountConditionalBaseline ----------


def test_count_conditional_predicts_per_count():
    """Build training data where each count state has a different mode."""
    rows = []
    rows += [{"balls": 0, "strikes": 0, "pitch_type_canonical": "FF"}] * 100
    rows += [{"balls": 0, "strikes": 0, "pitch_type_canonical": "SL"}] * 20
    rows += [{"balls": 3, "strikes": 2, "pitch_type_canonical": "FF"}] * 20
    rows += [{"balls": 3, "strikes": 2, "pitch_type_canonical": "SL"}] * 80
    df = pd.DataFrame(rows)

    cc = CountConditionalBaseline().fit(df)
    # 0-0 mode is FF; 3-2 mode is SL
    pred = cc.predict(pd.DataFrame({"balls": [0, 3], "strikes": [0, 2]}))
    assert pred[0] == PITCH_TYPE_TO_ID["FF"]
    assert pred[1] == PITCH_TYPE_TO_ID["SL"]


def test_count_conditional_proba_sums_to_one_per_row():
    rows = []
    for b in range(4):
        for s in range(3):
            rows += [{"balls": b, "strikes": s, "pitch_type_canonical": "FF"}] * 5
            rows += [{"balls": b, "strikes": s, "pitch_type_canonical": "SL"}] * 5
    df = pd.DataFrame(rows)
    cc = CountConditionalBaseline().fit(df)
    probs = cc.predict_proba(df)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


def test_count_conditional_falls_back_to_marginal_for_unseen_count():
    """A count state we never trained on should fall back to the marginal."""
    rows = [{"balls": 0, "strikes": 0, "pitch_type_canonical": "FF"}] * 100
    rows += [{"balls": 0, "strikes": 0, "pitch_type_canonical": "SL"}] * 50
    df = pd.DataFrame(rows)
    cc = CountConditionalBaseline().fit(df)

    # Predict for unseen count (3, 2) → fallback to marginal (FF dominant)
    pred = cc.predict(pd.DataFrame({"balls": [3], "strikes": [2]}))
    assert pred[0] == PITCH_TYPE_TO_ID["FF"]


def test_count_conditional_validates_required_columns():
    with pytest.raises(KeyError, match="missing columns"):
        CountConditionalBaseline().fit(pd.DataFrame({"pitch_type_canonical": ["FF"]}))
