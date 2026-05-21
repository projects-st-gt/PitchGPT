"""Rollout coherence: do(type=CU) must yield curveball-like velo."""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
from causal.nuisance import NuisanceModels
from causal.g_computation import g_compute

CKPT = Path("checkpoints_modal/tiny-fold0-v7/checkpoint_calibrated.pt")


def test_do_curveball_shifts_velo_slow():
    """Under do(type=CU), sampled velo should skew slower than do(type=FF).
    Requires a v7 (type_conditioned_heads) checkpoint."""
    if not CKPT.exists():
        import pytest; pytest.skip("v7 checkpoint not yet trained")
    nu = NuisanceModels(CKPT, device="cpu")
    val = pd.read_parquet("data/augmented/2024/2024-04-01.parquet")
    g = (val.sort_values(["game_pk", "at_bat_number", "pitch_number"])
            .groupby(["game_pk", "at_bat_number"]))
    ab = next(grp for _, grp in g if len(grp) >= 5).reset_index(drop=True)
    r_ff = g_compute(nu, ab, intervention_position=1, intervention_type="FF",
                     n_paths=300, rng_seed=0)
    r_cu = g_compute(nu, ab, intervention_position=1, intervention_type="CU",
                     n_paths=300, rng_seed=0)
    print(f"mean velo bin: FF={r_ff.intervention_velo_bin_mean:.2f} "
          f"CU={r_cu.intervention_velo_bin_mean:.2f}")
    assert r_cu.intervention_velo_bin_mean < r_ff.intervention_velo_bin_mean
