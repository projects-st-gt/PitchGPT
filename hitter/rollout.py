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
    plate_x: np.ndarray | None = None,
    plate_z: np.ndarray | None = None,
    velo_native: np.ndarray | None = None,
    spin_native: np.ndarray | None = None,
    spin_axis_sin: np.ndarray | None = None,
    spin_axis_cos: np.ndarray | None = None,
) -> pd.DataFrame:
    """Build the cascade's per-pitch feature frame for one rollout step (N paths).

    When ``plate_x``/``plate_z`` are provided (v8 MDN), uses real sampled
    coordinates instead of zone centroids.  When ``velo_native``/``spin_native``
    are provided, uses the model's native sampled values instead of per-type
    means.  Falls back to centroids/means when None (v7 compat).
    """
    n = len(type_ids)
    tid = np.asarray(type_ids, int)
    zid = np.asarray(zone_ids, int)
    if plate_x is not None and plate_z is not None:
        cx = np.asarray(plate_x, float)
        cz = np.asarray(plate_z, float)
    else:
        cx = np.array([centroids.get(int(z), [0.0, 2.5])[0] for z in zid], float)
        cz = np.array([centroids.get(int(z), [0.0, 2.5])[1] for z in zid], float)
    if velo_native is not None:
        velo = np.asarray(velo_native, float)
    else:
        velo = np.array([velo_by_type.get(int(t), _LEAGUE_VELO) for t in tid], float)
    if spin_native is not None:
        spin = np.asarray(spin_native, float)
    else:
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
        "spin_axis_sin": (np.asarray(spin_axis_sin, "float32") if spin_axis_sin is not None
                          else np.zeros(n, "float32")),
        "spin_axis_cos": (np.asarray(spin_axis_cos, "float32") if spin_axis_cos is not None
                          else np.zeros(n, "float32")),
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
    # Build profile columns INTO the dict before constructing the frame —
    # assigning them one-by-one afterward fragments the DataFrame (O(n²), slow
    # per cell, warning flood).
    bvec = np.asarray(batter_vec, "float32"); pvec = np.asarray(pitcher_stuff_vec, "float32")
    for i in range(len(bvec)):
        cols[f"b{i}"] = np.full(n, bvec[i], "float32")
    for i in range(len(pvec)):
        cols[f"p{i}"] = np.full(n, pvec[i], "float32")
    return pd.DataFrame(cols)


def make_hitter_step_fn(hitter_model, xwoba_to_outcome, *, batter_vec,
                        pitcher_stuff_vec, velo_by_type, spin_by_type,
                        same_hand, centroids, ctx_cat, foul_rate_fn):
    """Per-cell closure: maps a rollout step's sampled pitches -> (result_probs,
    outcome5). Composes build_step_features -> cascade.predict_cascade ->
    cascade_to_result_probs. ``foul_rate_fn(b, s)`` gives P(foul|contact).

    Returns ``step(type_ids, zone_ids, balls, strikes, prev_type_ids,
    prev_zone_ids, n_prev, **kwargs) -> (result_probs (N,7), outcome5 (N,5))``.

    Optional kwargs (v8 MDN path): ``plate_x``, ``plate_z``, ``velo_native``,
    ``spin_native``, ``spin_axis_sin``, ``spin_axis_cos``. When provided, these
    override zone centroids / per-type means / zero spin axis respectively.
    """
    def step(type_ids, zone_ids, balls, strikes, prev_type_ids, prev_zone_ids,
             n_prev, **kwargs):
        X = build_step_features(
            type_ids=type_ids, zone_ids=zone_ids, balls=balls, strikes=strikes,
            prev_type_ids=prev_type_ids, prev_zone_ids=prev_zone_ids, n_prev=n_prev,
            batter_vec=batter_vec, pitcher_stuff_vec=pitcher_stuff_vec,
            velo_by_type=velo_by_type, spin_by_type=spin_by_type,
            same_hand=same_hand, centroids=centroids, ctx_cat=ctx_cat,
            **kwargs)
        casc = hitter_model.predict_cascade(X)
        outcome5 = xwoba_to_outcome(casc["contact_quality"])      # (N,5)
        # foul rate is count-constant; rows in this step share (balls,strikes)?
        # not necessarily — apply per-row.
        fr = np.array([foul_rate_fn(int(b), int(s)) for b, s in zip(balls, strikes)])
        # cascade_to_result_probs takes a scalar foul_rate; fold per-row by looping
        # the unique rates is overkill — apply elementwise via the decomposition.
        rp = cascade_to_result_probs(casc["swing"], casc["called_strike"],
                                     casc["whiff"], 0.0, outcome5)  # foul=0 placeholder
        # re-apply per-row foul split: move contact*fr from in_play into foul
        s = np.clip(casc["swing"], 0, 1); w = np.clip(casc["whiff"], 0, 1)
        contact = s * (1 - w)
        in_play_total = rp[:, 4] + rp[:, 5] + rp[:, 6]            # currently all contact mass (foul=0)
        keep = 1.0 - fr                                           # fraction staying in play
        rp[:, 3] = contact * fr                                   # foul
        rp[:, 4] *= keep; rp[:, 5] *= keep; rp[:, 6] *= keep      # scale in-play down
        return rp, outcome5
    return step


# ---------------------------------------------------------------------------
# Matchup-card integration: load the cascade context once, build per-cell step fns
# ---------------------------------------------------------------------------

def load_hitter_ctx(model_dir: str = "checkpoints/hitter", fold_id: int = 0,
                    profiles_dir=None) -> dict:
    """Load everything a worker needs to build per-cell cascade step fns once:
    the HitterModel, the xwOBA→outcome map, zone centroids, and profile caches.
    ``profiles_dir`` overrides the ProfileCache location (e.g. on a Modal volume).
    """
    import json
    from pathlib import Path
    from data.profile_cache_loader import ProfileCache
    from hitter.model import HitterModel
    from hitter.compose import xwoba_outcome_fn
    pdir = Path(profiles_dir) if profiles_dir else None
    hm = HitterModel(model_dir)
    xfn = xwoba_outcome_fn(json.load(open(f"{model_dir}/xwoba_outcome_map.json")))
    cent = {int(k): v for k, v in
            json.load(open(f"{model_dir}/zone_centroids.json")).items()}
    return {
        "hm": hm, "xfn": xfn, "cent": cent,
        "bc": ProfileCache(role="batter", fold_id=fold_id, profiles_dir=pdir),
        "pc": ProfileCache(role="pitcher", fold_id=fold_id, profiles_dir=pdir),
    }


def build_cell_step_fn(ctx: dict, *, pitcher_id: int, batter_id: int,
                       stand: str, throws: str, game_date: str,
                       ballpark_id: int = 0, umpire_id: int = 0,
                       catcher_id: int = 0):
    """Build the hitter_step_fn for one matchup cell from a loaded ``ctx``.

    Pulls the batter & pitcher profiles as-of ``game_date`` (leakage-safe), derives
    the pitcher's per-type velo/spin means, and closes over the cascade.
    """
    import pandas as pd
    from data.profile_cache import PITCHER_FEATURE_INDEX as PFI
    from hitter.train import PITCHER_STUFF_FEATURES
    _PT = ("FF", "SI", "FC", "SL", "CU", "CH", "FS")

    asof = pd.Timestamp(game_date)
    bv = np.asarray(ctx["bc"].lookup(int(batter_id), asof, 1, as_of_fallback=True)["vector"], np.float32)
    pf = np.asarray(ctx["pc"].lookup(int(pitcher_id), asof, 1, as_of_fallback=True)["vector"], np.float32)
    pstuff = pf[[PFI[f] for f in PITCHER_STUFF_FEATURES]]
    velo = {j + 1: float(pf[PFI[f"mean_velo_{t}"]]) for j, t in enumerate(_PT)}
    spin = {j + 1: float(pf[PFI[f"mean_spin_{t}"]]) for j, t in enumerate(_PT)}
    return make_hitter_step_fn(
        ctx["hm"], ctx["xfn"], batter_vec=bv, pitcher_stuff_vec=pstuff,
        velo_by_type=velo, spin_by_type=spin,
        same_hand=int(str(stand) == str(throws)), centroids=ctx["cent"],
        ctx_cat={"umpire_id": int(umpire_id), "catcher_id": int(catcher_id),
                 "ballpark_id": int(ballpark_id), "roof_state": 1, "temp_bucket": 4},
        foul_rate_fn=ctx["hm"].foul_rate)
