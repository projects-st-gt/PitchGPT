"""Fractional in-play outcomes (Rao-Blackwellized terminal step) for
g_compute_v2 — synthetic micro-model fixtures (allowed in tests/).

Two properties:
  1. Unbiasedness: fractional and sampled outcome_dists agree (same paths,
     the in-play split is just credited exactly instead of sampled).
  2. Variance reduction: across seeds, the fractional outcome_dist varies
     LESS than the sampled one — the whole point of the feature.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from model.v2.config import V2Config
from model.v2.model import PitchGPTV2
from causal.g_computation_v2 import g_compute_v2
from causal.nuisance_v2 import NuisanceModelsV2


class _MicroNuisance:
    """Just enough of NuisanceModelsV2 for g_compute_v2 (no checkpoint I/O)."""

    def __init__(self):
        self.cfg = V2Config(
            n_layers=1, n_heads=2, d_model=16, d_ff=32, adaln_hidden=8,
            pitcher_profile_dim=5, batter_profile_dim=3, n_continuous=4,
        )
        torch.manual_seed(0)
        self.model = PitchGPTV2(self.cfg).eval()
        self.device = torch.device("cpu")
        self.temperatures = {"type": 1.0}
        self.count_temperatures = {}
        self.apply_temperatures = True
        # Profile stubs used by build_single_ab_batch_v2.
        self.pitcher_cache = SimpleNamespace(
            lookup=lambda *a, **k: {"vector": np.zeros(5, np.float32)})
        self.batter_cache = SimpleNamespace(
            lookup=lambda *a, **k: {"vector": np.zeros(3, np.float32)})
        self.standardizer = None

    forward = NuisanceModelsV2.forward
    predict_continuous = NuisanceModelsV2.predict_continuous
    scale_type_logits = NuisanceModelsV2.scale_type_logits
    _to_device = NuisanceModelsV2._to_device


def _stub_step_fn(tids, zids, balls, strikes, prev_t, prev_z, n_prev, **kw):
    """Fixed cascade: 30% ball, 20% called strike, 10% whiff, 10% foul,
    30% in-play with a non-degenerate 5-way split."""
    n = len(tids)
    rp = np.tile([0.30, 0.20, 0.10, 0.10, 0.18, 0.09, 0.03], (n, 1))
    oc5 = np.tile([0.60, 0.22, 0.08, 0.01, 0.09], (n, 1))
    return rp, oc5


def _one_pitch_ab():
    return pd.DataFrame([{
        "game_date": "2024-08-01", "pitcher": 1, "batter": 2,
        "pitch_number": 1, "type_id": 1, "result_id": 1,
        "count_state": 0, "outs_state": 0, "runners_state": 0,
        "release_speed": 92.0, "release_spin_rate": 2200.0,
        "plate_x": 0.1, "plate_z": 2.5,
    }])


def _dist(nz, *, fractional, seed):
    r = g_compute_v2(nz, _one_pitch_ab(), n_paths=300, rng_seed=seed,
                     hitter_step_fn=_stub_step_fn,
                     fractional_inplay=fractional)
    return r.ab_outcome_distribution


def test_fractional_matches_sampled_in_expectation():
    nz = _MicroNuisance()
    torch.manual_seed(1)
    frac = np.mean([_dist(nz, fractional=True, seed=s) for s in range(6)], axis=0)
    torch.manual_seed(1)
    samp = np.mean([_dist(nz, fractional=False, seed=s) for s in range(6)], axis=0)
    assert np.all(np.isfinite(frac)) and abs(frac.sum() - 1.0) < 1e-6
    # Same generator seeds -> same paths; only the terminal credit differs.
    # Agreement within MC tolerance on every class.
    assert np.allclose(frac, samp, atol=0.03), (
        f"fractional {np.round(frac,3)} vs sampled {np.round(samp,3)}")


def test_fractional_reduces_variance():
    nz = _MicroNuisance()
    torch.manual_seed(1)
    frac = np.array([_dist(nz, fractional=True, seed=s) for s in range(8)])
    torch.manual_seed(1)
    samp = np.array([_dist(nz, fractional=False, seed=s) for s in range(8)])
    # Variance of the in-play classes (1B idx 2 .. out idx 6 in AB order
    # K,BB,1B,2B,3B,HR,out) must shrink; compare summed std over classes.
    frac_std = frac.std(axis=0).sum()
    samp_std = samp.std(axis=0).sum()
    assert frac_std < samp_std, (
        f"fractional summed std {frac_std:.4f} must be < sampled {samp_std:.4f}")


def test_sampled_mode_unchanged_semantics():
    # outcome_dist must still sum to 1 over valid paths and run_value finite.
    nz = _MicroNuisance()
    r = g_compute_v2(nz, _one_pitch_ab(), n_paths=200, rng_seed=3,
                     hitter_step_fn=_stub_step_fn, fractional_inplay=False)
    assert abs(r.ab_outcome_distribution.sum() - 1.0) < 1e-6
    assert np.isfinite(r.mean_run_value)
