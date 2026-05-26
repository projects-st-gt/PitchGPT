"""Re-runnable v6/v7 prediction eval driver.

Calls scripts.eval_held_out_pitchers for v6 and v7, then scripts.eval_postseason_ood
for the side-by-side. CPU-forced (MPS miscompiles the ab_outcome gather).

Outputs go to whichever stdout the caller redirects; this script just runs the
three jobs sequentially.
"""
import torch
torch.backends.mps.is_available = lambda: False  # MPS miscompiles the ab_outcome gather
import runpy, sys, traceback

jobs = [
    ("scripts.eval_held_out_pitchers",
     ["--ckpt", "checkpoints_modal/tiny-fold0-v6/checkpoint_calibrated.pt"]),
    ("scripts.eval_held_out_pitchers",
     ["--ckpt", "checkpoints_modal/tiny-fold0-v7/checkpoint_calibrated.pt"]),
    ("scripts.eval_postseason_ood",
     ["--ckpt", "checkpoints_modal/tiny-fold0-v6/checkpoint_calibrated.pt",
      "--ckpt", "checkpoints_modal/tiny-fold0-v7/checkpoint_calibrated.pt"]),
]
for mod, args in jobs:
    print(f"\n\n========== {mod} {args} ==========", flush=True)
    sys.argv = [mod] + args
    try:
        runpy.run_module(mod, run_name="__main__")
    except SystemExit:
        pass
    except Exception:
        traceback.print_exc()
print("\n\n=== ALL PREDICTION EVALS DONE ===", flush=True)
