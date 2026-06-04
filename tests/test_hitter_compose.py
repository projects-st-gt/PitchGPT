"""Tests for hitter.compose — the analytic count-tree solve.

The count is a small absorbing Markov chain (12 ball-strike states + 7 terminals).
Pure-math pieces (transition matrix build, fundamental-matrix solve, per-PA rate
formulas) are tested on degenerate transition distributions with hand-computable
answers. cascade_transition is tested with a synthetic hitter model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hitter.compose import (
    COUNTS, TERMINALS, count_index,
    build_transition_matrix, solve_terminal_distribution,
    per_pa_outcome, cascade_transition, compose_pa,
    build_xwoba_outcome_map, xwoba_outcome_fn,
)


def _const_transition(**event_probs):
    """A transition dict identical for every count (each must sum to 1)."""
    base = {"ball": 0.0, "strike": 0.0, "stay": 0.0,
            "out": 0.0, "1B": 0.0, "2B": 0.0, "3B": 0.0, "HR": 0.0}
    base.update(event_probs)
    assert abs(sum(base.values()) - 1.0) < 1e-9
    return {c: dict(base) for c in COUNTS}


def test_counts_and_terminals_shapes():
    assert len(COUNTS) == 12                       # balls 0-3 x strikes 0-2
    assert TERMINALS == ["BB", "K", "out", "1B", "2B", "3B", "HR"]
    assert count_index((0, 0)) == 0


def test_all_called_strikes_gives_strikeout():
    """Every pitch a strike -> PA always ends in K."""
    trans = _const_transition(strike=1.0)
    Q, R = build_transition_matrix(trans)
    term = solve_terminal_distribution(Q, R, start=(0, 0))
    assert term["K"] == pytest.approx(1.0)
    assert sum(term.values()) == pytest.approx(1.0)


def test_all_balls_gives_walk():
    """Every pitch a ball -> PA always ends in BB."""
    trans = _const_transition(ball=1.0)
    Q, R = build_transition_matrix(trans)
    term = solve_terminal_distribution(Q, R, start=(0, 0))
    assert term["BB"] == pytest.approx(1.0)


def test_all_in_play_out_gives_out():
    trans = _const_transition(out=1.0)
    Q, R = build_transition_matrix(trans)
    term = solve_terminal_distribution(Q, R, start=(0, 0))
    assert term["out"] == pytest.approx(1.0)


def test_two_strike_foul_self_loop_terminates():
    """A pitch that's 70% foul / 30% strike still terminates in K (the foul
    self-loops at 2 strikes but eventually a strike lands)."""
    trans = _const_transition(strike=0.3, stay=0.7)
    Q, R = build_transition_matrix(trans)
    term = solve_terminal_distribution(Q, R, start=(0, 0))
    # at <2 strikes 'stay' is impossible, but the const dict applies everywhere;
    # build_transition_matrix must treat 'stay' as self-loop only where s==2 and
    # redirect it to a strike where s<2 (a foul advances the count pre-2-strikes).
    assert term["K"] == pytest.approx(1.0)


def test_per_pa_outcome_formulas():
    # half hits (all 1B), half outs -> AVG .500, OBP .500, SLG .500, OPS 1.000
    term = {"BB": 0.0, "K": 0.0, "out": 0.5, "1B": 0.5,
            "2B": 0.0, "3B": 0.0, "HR": 0.0}
    m = per_pa_outcome(term)
    assert m["AVG"] == pytest.approx(0.5)
    assert m["OBP"] == pytest.approx(0.5)
    assert m["SLG"] == pytest.approx(0.5)
    assert m["OPS"] == pytest.approx(1.0)
    assert m["K_pct"] == pytest.approx(0.0)
    assert m["BB_pct"] == pytest.approx(0.0)


def test_per_pa_outcome_walk_lifts_obp_not_avg():
    term = {"BB": 0.2, "K": 0.0, "out": 0.4, "1B": 0.4,
            "2B": 0.0, "3B": 0.0, "HR": 0.0}
    m = per_pa_outcome(term)
    # AVG = hits/AB = 0.4 / (1 - 0.2) = 0.5 ; OBP = hits+BB = 0.6
    assert m["AVG"] == pytest.approx(0.5)
    assert m["OBP"] == pytest.approx(0.6)
    assert m["BB_pct"] == pytest.approx(0.2)


# ---- cascade_transition with a synthetic hitter model -----------------------

class _FakeHitter:
    """Deterministic cascade outputs for testing the marginalization math."""
    def __init__(self, swing, whiff, called, xwoba):
        self._s, self._w, self._c, self._x = swing, whiff, called, xwoba

    def predict_cascade(self, X):
        n = len(X)
        return {"swing": np.full(n, self._s), "whiff": np.full(n, self._w),
                "called_strike": np.full(n, self._c),
                "contact_quality": np.full(n, self._x)}

    def foul_rate(self, b, s):
        return 0.5


def _ident_xwoba_to_outcome(xwoba):
    """Map any xwOBA -> always a single (1B)."""
    n = len(xwoba)
    out = np.zeros((n, 5))
    out[:, 1] = 1.0    # column order [out,1B,2B,3B,HR]
    return out


def test_cascade_transition_pure_take_called_strike():
    """swing=0, called=1 -> every pitch is a called strike."""
    hm = _FakeHitter(swing=0.0, whiff=0.0, called=1.0, xwoba=0.3)
    pitches = pd.DataFrame({"f": [0.0, 0.0]})
    w = np.array([0.5, 0.5])
    t = cascade_transition((0, 0), pitches, w, hm, _ident_xwoba_to_outcome)
    assert t["strike"] == pytest.approx(1.0)
    assert t["ball"] == pytest.approx(0.0)


def test_cascade_transition_swing_contact_fair_inplay():
    """swing=1, whiff=0, foul=0.5 -> half fair (in play -> 1B), half foul."""
    hm = _FakeHitter(swing=1.0, whiff=0.0, called=0.0, xwoba=0.3)
    pitches = pd.DataFrame({"f": [0.0]})
    t = cascade_transition((0, 0), pitches, np.array([1.0]), hm,
                           _ident_xwoba_to_outcome)
    # in-play prob = swing*(1-whiff)*(1-foul) = 0.5, all -> 1B
    assert t["1B"] == pytest.approx(0.5)
    # foul at 0-0 (s<2) advances count -> strike
    assert t["strike"] == pytest.approx(0.5)


def test_xwoba_outcome_map_low_to_out_high_to_hr():
    """Empirical map: balls with low xwOBA were outs, high xwOBA were HR."""
    n = 2000
    xw = np.linspace(0.0, 2.0, n)
    events = np.where(xw < 0.5, "field_out", "home_run")
    m = build_xwoba_outcome_map(xw, events, n_bins=10)
    fn = xwoba_outcome_fn(m)
    lo = fn(np.array([0.1]))[0]      # [out,1B,2B,3B,HR]
    hi = fn(np.array([1.9]))[0]
    assert lo[0] > 0.9               # mostly out
    assert hi[4] > 0.9               # mostly HR
    # every row is a valid distribution
    assert np.allclose(fn(np.array([0.1, 1.9])).sum(axis=1), 1.0)


def test_compose_pa_end_to_end_with_fake_pitch_provider():
    """compose_pa wires pitch-provider + cascade + solve -> per-PA metrics."""
    hm = _FakeHitter(swing=0.5, whiff=0.3, called=0.6, xwoba=0.35)

    def pitch_provider(count):
        return pd.DataFrame({"f": [0.0, 0.0]}), np.array([0.5, 0.5])

    m = compose_pa(hm, pitch_provider, _ident_xwoba_to_outcome)
    # sanity: it returns a valid metric bundle with OPS in a plausible range
    assert set(m) >= {"AVG", "OBP", "SLG", "OPS", "K_pct", "BB_pct"}
    assert 0.0 <= m["OPS"] <= 5.0
    assert 0.0 <= m["K_pct"] <= 1.0
    print(f"\ncompose_pa fake: OPS={m['OPS']:.3f} K%={m['K_pct']:.3f} "
          f"BB%={m['BB_pct']:.3f} AVG={m['AVG']:.3f}")
