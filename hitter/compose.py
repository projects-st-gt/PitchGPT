"""Analytic count-tree composition (hitter/MODEL_DESIGN.md §3).

The plate appearance is a small **absorbing Markov chain**: 12 ball-strike count
states + 7 terminals {BB, K, out, 1B, 2B, 3B, HR}. For each count, the pitch
model (PitchGPT, π̂) supplies a distribution over candidate pitches and the hitter
cascade supplies the per-pitch response; marginalizing pitch choice gives a per-
count transition distribution. We build the transition matrix and **solve for the
terminal distribution in closed form** (fundamental matrix) — zero Monte-Carlo
noise, instant, smooth. From the terminal distribution we read per-PA
AVG/OBP/SLG/OPS/K%/BB%.

The pure-math core (build_transition_matrix / solve_terminal_distribution /
per_pa_outcome) is independent of PitchGPT and the trained model; ``compose_pa``
wires a pitch-distribution provider + a HitterModel together.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

#: Transient count states (balls 0-3 x strikes 0-2), fixed order.
COUNTS: list[tuple[int, int]] = [(b, s) for b in range(4) for s in range(3)]
_COUNT_IDX = {c: i for i, c in enumerate(COUNTS)}
#: Absorbing terminal states, fixed order.
TERMINALS: list[str] = ["BB", "K", "out", "1B", "2B", "3B", "HR"]
_TERM_IDX = {t: i for i, t in enumerate(TERMINALS)}
_INPLAY = ["out", "1B", "2B", "3B", "HR"]   # xwoba_to_outcome column order


def count_index(count: tuple[int, int]) -> int:
    return _COUNT_IDX[count]


def build_transition_matrix(
    transitions: dict[tuple[int, int], dict[str, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Build (Q, R): transient->transient (12x12) and transient->absorbing (12x7).

    ``transitions[count]`` is a per-pitch event distribution over
    {ball, strike, stay, out, 1B, 2B, 3B, HR} (each summing to 1). Count routing:
    ball -> (b+1, s) or BB at b+1==4; strike -> (b, s+1) or K at s+1==3;
    'stay' is the 2-strike-foul self-loop, but at s<2 a foul advances the count,
    so 'stay' mass there is routed as a strike.
    """
    n = len(COUNTS)
    Q = np.zeros((n, n))
    R = np.zeros((n, len(TERMINALS)))

    def add_strike(i, b, s, prob):
        if s + 1 == 3:
            R[i, _TERM_IDX["K"]] += prob
        else:
            Q[i, _COUNT_IDX[(b, s + 1)]] += prob

    def add_ball(i, b, s, prob):
        if b + 1 == 4:
            R[i, _TERM_IDX["BB"]] += prob
        else:
            Q[i, _COUNT_IDX[(b + 1, s)]] += prob

    for (b, s), events in transitions.items():
        i = _COUNT_IDX[(b, s)]
        for event, prob in events.items():
            if prob == 0:
                continue
            if event == "ball":
                add_ball(i, b, s, prob)
            elif event == "strike":
                add_strike(i, b, s, prob)
            elif event == "stay":
                if s < 2:                       # foul before 2 strikes -> strike
                    add_strike(i, b, s, prob)
                else:                            # 2-strike foul -> self-loop
                    Q[i, i] += prob
            else:                                # in-play absorbing outcome
                R[i, _TERM_IDX[event]] += prob
    return Q, R


def solve_terminal_distribution(
    Q: np.ndarray, R: np.ndarray, start: tuple[int, int] = (0, 0),
) -> dict[str, float]:
    """Closed-form absorption distribution from ``start`` via the fundamental
    matrix N = (I - Q)^{-1}; B = N R gives P(absorb in each terminal | start)."""
    n = Q.shape[0]
    N = np.linalg.inv(np.eye(n) - Q)
    B = N @ R
    row = B[_COUNT_IDX[start]]
    return {t: float(row[_TERM_IDX[t]]) for t in TERMINALS}


def per_pa_outcome(terminal: dict[str, float]) -> dict[str, float]:
    """Per-PA AVG/OBP/SLG/OPS/K%/BB% from the terminal distribution.

    AB fraction = 1 - BB (HBP/SF ignored in v0). AVG = hits/AB; OBP = hits + BB
    (PA=1); SLG = total bases / AB; OPS = OBP + SLG.
    """
    bb = terminal["BB"]
    hits = terminal["1B"] + terminal["2B"] + terminal["3B"] + terminal["HR"]
    tb = (terminal["1B"] + 2 * terminal["2B"] + 3 * terminal["3B"]
          + 4 * terminal["HR"])
    ab = 1.0 - bb
    avg = hits / ab if ab > 1e-12 else 0.0
    slg = tb / ab if ab > 1e-12 else 0.0
    obp = hits + bb
    return {"AVG": avg, "OBP": obp, "SLG": slg, "OPS": obp + slg,
            "K_pct": terminal["K"], "BB_pct": bb}


def cascade_transition(
    count: tuple[int, int],
    pitch_features: pd.DataFrame,
    pitch_weights: np.ndarray,
    hitter,
    xwoba_to_outcome,
) -> dict[str, float]:
    """Marginal per-pitch transition distribution for ``count``.

    Runs the hitter cascade on the candidate pitches, forms each pitch's event
    distribution {ball, strike, stay, out..HR}, then averages over pitches by
    ``pitch_weights`` (π̂(pitch | count)). ``xwoba_to_outcome(xwoba)`` maps the
    contact-quality regression output to a (n, 5) distribution over
    [out, 1B, 2B, 3B, HR].
    """
    b, s = count
    casc = hitter.predict_cascade(pitch_features)
    p_swing = np.clip(casc["swing"], 0, 1)
    p_whiff = np.clip(casc["whiff"], 0, 1)
    p_called = np.clip(casc["called_strike"], 0, 1)
    xwoba = casc["contact_quality"]
    fr = float(hitter.foul_rate(b, s))

    p_take = 1.0 - p_swing
    p_ball = p_take * (1.0 - p_called)
    p_cs = p_take * p_called
    p_contact = p_swing * (1.0 - p_whiff)
    p_whiff_strike = p_swing * p_whiff
    p_foul = p_contact * fr
    p_fair = p_contact * (1.0 - fr)

    strike_mass = p_cs + p_whiff_strike + (p_foul if s < 2 else 0.0)
    stay_mass = p_foul if s == 2 else np.zeros_like(p_foul)
    ball_mass = p_ball

    oc = xwoba_to_outcome(xwoba)                      # (n, 5)
    inplay_split = p_fair[:, None] * oc               # (n, 5)

    w = np.asarray(pitch_weights, dtype=float)
    w = w / w.sum()
    t = {"ball": float(w @ ball_mass),
         "strike": float(w @ strike_mass),
         "stay": float(w @ np.broadcast_to(stay_mass, p_foul.shape))}
    agg = (w[:, None] * inplay_split).sum(axis=0)     # (5,)
    for k, name in enumerate(_INPLAY):
        t[name] = float(agg[k])
    return t


def compose_pa(
    hitter,
    pitch_provider,
    xwoba_to_outcome,
    *,
    start: tuple[int, int] = (0, 0),
) -> dict[str, float]:
    """Full analytic per-PA outcome for one matchup.

    ``pitch_provider(count) -> (pitch_features_df, weights)`` supplies π̂'s pitch
    distribution per count (the PitchGPT seam). Returns the per-PA metric bundle
    plus the raw terminal distribution under key ``"terminal"``.
    """
    transitions = {}
    for count in COUNTS:
        feats, weights = pitch_provider(count)
        transitions[count] = cascade_transition(
            count, feats, weights, hitter, xwoba_to_outcome)
    Q, R = build_transition_matrix(transitions)
    terminal = solve_terminal_distribution(Q, R, start=start)
    metrics = per_pa_outcome(terminal)
    metrics["terminal"] = terminal
    return metrics
