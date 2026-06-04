"""Bridge the hitter cascade into the pitchGPT Monte-Carlo rollout (g_compute).

The rollout already samples pitches from pitchGPT (π̂) sequentially — the count
and the full pitch sequence emerge naturally. This module replaces the *outcome*
step: instead of the transformer's weak result head, the **cascade** decides the
batter's response to each sampled pitch. ``cascade_to_result_probs`` is the
translator from cascade outputs to the simulator's 7-class per-pitch result vocab,
so the existing count machine + termination logic run unchanged.
"""
from __future__ import annotations

import numpy as np

#: The simulator's per-pitch result vocabulary, in model-index order
#: (data/dataset RESULT_TO_ID). cascade_to_result_probs emits columns in THIS order.
RESULT_ORDER = ["ball", "called_strike", "swinging_strike", "foul",
                "in_play_out", "in_play_hit", "in_play_hr"]


def cascade_to_result_probs(
    p_swing: np.ndarray,
    p_called_strike: np.ndarray,
    p_whiff: np.ndarray,
    foul_rate: float,
    outcome5: np.ndarray,
) -> np.ndarray:
    """Translate per-pitch cascade outputs -> (N, 7) result distribution.

    Inputs (each (N,) except foul_rate scalar and outcome5 (N,5)):
    - ``p_swing``         P(swing)
    - ``p_called_strike`` P(called strike | take)
    - ``p_whiff``         P(whiff | swing)
    - ``foul_rate``       P(foul | contact) for the current count (v0 constant)
    - ``outcome5``        (N,5) P over [out, 1B, 2B, 3B, HR] for balls in play

    The pitch's event probabilities decompose as:
        ball            = (1-swing)(1-cs)
        called_strike   = (1-swing)(cs)
        swinging_strike = swing·whiff
        foul            = swing(1-whiff)·foul_rate
        in_play         = swing(1-whiff)(1-foul_rate), split by outcome5 into
                          in_play_hr = P(HR), in_play_hit = P(1B+2B+3B),
                          in_play_out = P(out).
    Columns are returned in ``RESULT_ORDER`` and sum to 1 per row.
    """
    s = np.clip(np.asarray(p_swing, float), 0, 1)
    cs = np.clip(np.asarray(p_called_strike, float), 0, 1)
    w = np.clip(np.asarray(p_whiff, float), 0, 1)
    oc = np.asarray(outcome5, float)                      # (N,5) [out,1B,2B,3B,HR]

    take = 1.0 - s
    contact = s * (1.0 - w)
    in_play = contact * (1.0 - foul_rate)

    n = len(s)
    r = np.zeros((n, 7))
    r[:, 0] = take * (1.0 - cs)                            # ball
    r[:, 1] = take * cs                                    # called_strike
    r[:, 2] = s * w                                        # swinging_strike
    r[:, 3] = contact * foul_rate                          # foul
    r[:, 4] = in_play * oc[:, 0]                           # in_play_out
    r[:, 5] = in_play * (oc[:, 1] + oc[:, 2] + oc[:, 3])   # in_play_hit (1B+2B+3B)
    r[:, 6] = in_play * oc[:, 4]                           # in_play_hr
    return r
