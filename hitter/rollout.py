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
import pandas as pd

#: feature_zone ids 0-8 are inside the strike zone (3x3 grid); 9-12 are the
#: outside-corner zones. Used to set in_zone from a sampled zone.
_IN_ZONE_IDS = set(range(9))
_LEAGUE_VELO, _LEAGUE_SPIN = 89.0, 2200.0   # fallbacks for a type with no profile mean

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


def build_step_features(
    *,
    type_ids: np.ndarray,
    zone_ids: np.ndarray,
    balls: np.ndarray,
    strikes: np.ndarray,
    prev_type_ids: np.ndarray,
    prev_zone_ids: np.ndarray,
    n_prev: np.ndarray,
    batter_vec: np.ndarray,
    pitcher_stuff_vec: np.ndarray,
    velo_by_type: dict,
    spin_by_type: dict,
    same_hand: int,
    centroids: dict,
    ctx_cat: dict,
) -> pd.DataFrame:
    """Build the cascade's per-pitch feature frame for one rollout step (N paths).

    Pitch identity (type, zone) comes from what pitchGPT just sampled; location is
    the zone centroid, velo/spin the pitcher's per-type means (the sampled
    velo-bin is a deviation decile ≈ the type mean, so we use the mean and skip
    the bin→mph inversion). ``prev_type_ids`` carries the SEQUENCE (the pitch
    before), so the cascade's lag features are real inside the rollout. Context
    columns (outs, spin_axis) use neutral values — secondary to type/location/
    count/profiles. Batter & pitcher profile vectors broadcast across rows.
    """
    n = len(type_ids)
    tid = np.asarray(type_ids, int)
    zid = np.asarray(zone_ids, int)
    cx = np.array([centroids.get(int(z), [0.0, 2.5])[0] for z in zid], float)
    cz = np.array([centroids.get(int(z), [0.0, 2.5])[1] for z in zid], float)
    velo = np.array([velo_by_type.get(int(t), _LEAGUE_VELO) for t in tid], float)
    spin = np.array([spin_by_type.get(int(t), _LEAGUE_SPIN) for t in tid], float)
    in_zone = np.isin(zid, list(_IN_ZONE_IDS)).astype("int8")
    prev_in_zone = np.where(
        prev_zone_ids < 0, -1,
        np.isin(np.asarray(prev_zone_ids, int), list(_IN_ZONE_IDS)).astype(int),
    ).astype("int8")

    cols = {
        "type_id": tid.astype("int16"),
        "plate_x": cx.astype("float32"), "plate_z": cz.astype("float32"),
        "release_speed": velo.astype("float32"),
        "release_spin_rate": spin.astype("float32"),
        "spin_axis_sin": np.zeros(n, "float32"),
        "spin_axis_cos": np.zeros(n, "float32"),
        "balls": np.asarray(balls, "int16"), "strikes": np.asarray(strikes, "int16"),
        "pitch_number": (np.asarray(n_prev, int) + 1).astype("int16"),
        "in_zone": in_zone,
        "prev_type_id": np.asarray(prev_type_ids, "int16"),
        "prev_in_zone": prev_in_zone,
        "n_prev_pitches": np.asarray(n_prev, "int16"),
        "same_hand": np.full(n, int(same_hand), "int8"),
        "outs_when_up": np.zeros(n, "int16"),
    }
    for name, val in ctx_cat.items():
        cols[name] = np.full(n, int(val), "int32")
    df = pd.DataFrame(cols)
    bvec = np.asarray(batter_vec, "float32"); pvec = np.asarray(pitcher_stuff_vec, "float32")
    for i in range(len(bvec)):
        df[f"b{i}"] = bvec[i]
    for i in range(len(pvec)):
        df[f"p{i}"] = pvec[i]
    return df
