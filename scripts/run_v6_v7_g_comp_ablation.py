"""v6 vs v7 g-computation ablation (ADR 013 closeout for Part 1).

ADR 013 mandates: "run the population AIPW / a sample of g-computation queries
on the v6 (incoherent) vs v7 (coherent) rollout and report the delta."

This script does the g-computation sample. For each of N_ABS test-split at-bats:
  - run g_compute under {v6, v7} × {do(FF), do(CU)} at intervention_position=1
  - record mean_run_value, SE, and the mean sampled velo bin at the intervention
Then aggregate:
  - per-version effect τ̂(CU vs FF) — the run-value contrast
  - v7 − v6 delta (the ADR-013 ablation number)
  - "velo coherence" diagnostic: mean(velo|do(CU)) − mean(velo|do(FF))
    Under v6 (incoherent execution heads) this should be ~0 — the velo head
    ignores the intervened type. Under v7 (coherent) it should be substantially
    negative — CU velo bin is genuinely lower than FF velo bin.

CPU-forced (MPS miscompiles the ab_outcome gather; reproduces on v6 too,
unrelated to v7).
"""
from __future__ import annotations

import torch
torch.backends.mps.is_available = lambda: False  # MPS miscompiles ab_outcome gather

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from causal.nuisance import NuisanceModels
from causal.g_computation import g_compute

V6 = Path("checkpoints_modal/tiny-fold0-v6/checkpoint_calibrated.pt")
V7 = Path("checkpoints_modal/tiny-fold0-v7/checkpoint_calibrated.pt")
OUT_DIR = Path("eval/results/v7-closeout")

N_ABS = 20
N_PATHS = 500   # < default 1000 for runtime; still enough to read off coherence signal
INTERVENTIONS = ["FF", "CU"]
INTERVENTION_POSITION = 1
SEED = 42


def load_test_split() -> pd.DataFrame:
    """Test split per CLAUDE.md hard rule 2: 2024 H2 + 2025."""
    files = sorted(Path("data/augmented/2024").glob("2024-*.parquet"))
    files = [f for f in files if f.stem >= "2024-07-16"]
    files += sorted(Path("data/augmented/2025").glob("2025-*.parquet"))
    # subset for speed — first few days is plenty to draw N_ABS at-bats from
    files = files[:6]
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values(["game_pk", "at_bat_number", "pitch_number"]).reset_index(drop=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = load_test_split()
    print(f"loaded {len(df):,} pitches from test split", flush=True)
    ab_groups = list(df.groupby(["game_pk", "at_bat_number"], sort=False))
    # require >= 3 pitches: pos=1 is the 2nd pitch, leave room to roll out
    ab_groups = [(k, g) for k, g in ab_groups if len(g) >= 3]
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(ab_groups), size=min(N_ABS, len(ab_groups)), replace=False)
    chosen = [ab_groups[int(i)] for i in idx]
    print(f"sampled {len(chosen)} at-bats (>=3 pitches)", flush=True)

    v6 = NuisanceModels(V6, device="cpu")
    v7 = NuisanceModels(V7, device="cpu")
    print("loaded v6 + v7 nuisance models", flush=True)

    rows = []
    for i, (ab_key, ab) in enumerate(chosen):
        ab = ab.reset_index(drop=True)
        out = {
            "game_pk": int(ab_key[0]),
            "at_bat_number": int(ab_key[1]),
            "ab_len": int(len(ab)),
        }
        for ver, nu in [("v6", v6), ("v7", v7)]:
            for itype in INTERVENTIONS:
                r = g_compute(
                    nu, ab,
                    intervention_position=INTERVENTION_POSITION,
                    intervention_type=itype,
                    n_paths=N_PATHS,
                    rng_seed=SEED,
                )
                out[f"{ver}_{itype}_mean_rv"] = float(r.mean_run_value)
                out[f"{ver}_{itype}_se_rv"] = float(r.se_run_value)
                out[f"{ver}_{itype}_velo_bin"] = float(r.intervention_velo_bin_mean)
                out[f"{ver}_{itype}_n_truncated"] = int(r.n_truncated)
        out["v6_effect"] = out["v6_CU_mean_rv"] - out["v6_FF_mean_rv"]
        out["v7_effect"] = out["v7_CU_mean_rv"] - out["v7_FF_mean_rv"]
        out["delta"] = out["v7_effect"] - out["v6_effect"]
        out["v6_velo_shift"] = out["v6_CU_velo_bin"] - out["v6_FF_velo_bin"]
        out["v7_velo_shift"] = out["v7_CU_velo_bin"] - out["v7_FF_velo_bin"]
        rows.append(out)
        print(
            f"[{i+1:>2}/{len(chosen)}] gpk={out['game_pk']} AB={out['at_bat_number']} "
            f"len={out['ab_len']}  v6 τ={out['v6_effect']:+.3f}  v7 τ={out['v7_effect']:+.3f}  "
            f"Δ={out['delta']:+.3f}  velo-shift v6={out['v6_velo_shift']:+.2f} "
            f"v7={out['v7_velo_shift']:+.2f}",
            flush=True,
        )

    df_out = pd.DataFrame(rows)
    csv_path = OUT_DIR / "g_comp_ablation.csv"
    df_out.to_csv(csv_path, index=False)

    summary = {
        "n_abs": int(len(df_out)),
        "n_paths_per_call": N_PATHS,
        "intervention_position": INTERVENTION_POSITION,
        "v6_mean_effect_CU_vs_FF": float(df_out["v6_effect"].mean()),
        "v6_std_effect_across_abs": float(df_out["v6_effect"].std()),
        "v7_mean_effect_CU_vs_FF": float(df_out["v7_effect"].mean()),
        "v7_std_effect_across_abs": float(df_out["v7_effect"].std()),
        "mean_delta_v7_minus_v6": float(df_out["delta"].mean()),
        "v6_mean_velo_shift_CU_minus_FF": float(df_out["v6_velo_shift"].mean()),
        "v7_mean_velo_shift_CU_minus_FF": float(df_out["v7_velo_shift"].mean()),
        "wallclock_s": round(time.time() - t0, 1),
    }
    (OUT_DIR / "g_comp_ablation_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n=== Aggregates ===")
    print(f"n_ABs:                       {summary['n_abs']}")
    print(f"v6 τ̂(CU vs FF):              {summary['v6_mean_effect_CU_vs_FF']:+.4f}  (SD {summary['v6_std_effect_across_abs']:.4f})")
    print(f"v7 τ̂(CU vs FF):              {summary['v7_mean_effect_CU_vs_FF']:+.4f}  (SD {summary['v7_std_effect_across_abs']:.4f})")
    print(f"mean Δ (v7 − v6):            {summary['mean_delta_v7_minus_v6']:+.4f}")
    print(f"v6 velo-shift (CU−FF) mean:  {summary['v6_mean_velo_shift_CU_minus_FF']:+.3f}  ← ~0 expected (incoherent)")
    print(f"v7 velo-shift (CU−FF) mean:  {summary['v7_mean_velo_shift_CU_minus_FF']:+.3f}  ← negative expected (coherent)")
    print(f"\nsaved -> {csv_path}")
    print(f"saved -> {OUT_DIR / 'g_comp_ablation_summary.json'}")
    print(f"wallclock: {summary['wallclock_s']}s")


if __name__ == "__main__":
    main()
