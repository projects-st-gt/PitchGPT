"""PitchGPT model configuration.

All architectural knobs live here. The Tiny/Small/Sanity factories below
match the locked-in plan from the pitchgpt-model skill and the
architecture brainstorm. Do not change vocab sizes here without bumping
the corresponding embedding-table parameters everywhere downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PitchGPTConfig:
    """Architecture spec for one PitchGPT model.

    Fields are grouped: trunk, factor vocabs, context vocabs, head sizes.
    The defaults are for **Tiny**; use the factory helpers below for other
    sizes.
    """

    # --- Trunk ---
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    d_ff: int = 1024  # 4 × d_model (FFN expansion factor 4)
    dropout: float = 0.1
    init_std: float = 0.02

    # --- Factor vocabularies (per pitchgpt-model skill) ---
    n_pitch_types: int = 8       # 7 canonical + PAD
    n_zones: int = 13            # SIS 14-zone scheme: 9 in-zone + 4 OOZ quadrants
                                 # (13 actual indices; no PAD slot since the
                                 # preprocess step drops pitches with NaN zone)
    n_velo_bins: int = 11        # 10 type-relative deciles + MISSING
    n_spin_rate_bins: int = 9    # 8 bins + MISSING
    n_spin_axis_bins: int = 13   # used only if not circular; 12 bins + MISSING
    n_result_classes: int = 8    # 7 result classes + PAD
    n_count_states: int = 12     # balls × strikes
    n_runner_states: int = 8     # 8 base-occupancy patterns
    n_outs: int = 3              # 0, 1, 2
    n_positions: int = 15        # pitch index within at-bat (max ~12, +buffer)

    # Spin axis: circular (sin/cos continuous) vs categorical
    # The architecture brainstorm argued for circular; that's the default.
    spin_axis_circular: bool = True

    # Inject the (standardized) pitcher & batter profile projections directly
    # into every pitch token's embedding, in addition to the prepended context
    # tokens. Tested 2026-05-11: this HURT vs. standardization-only at 6K steps
    # (0.442 vs 0.466 type top-1) — likely it dilutes the factored pitch-token
    # signal with redundant info already available via the context tokens.
    # Kept as a flag for future ablation; default OFF.
    inject_profiles_per_pitch: bool = False

    # Inject the (standardized) pitcher & batter profile projections directly
    # into the PROPENSITY head's hidden input (bypasses the trunk entirely, so
    # the head doesn't depend on attention back to the context tokens to fetch
    # the player fingerprint — mirrors the LSTM's profile-initialized state).
    # Tested 2026-05-11: this HURT slightly (-1pp vs std-only); the trunk
    # already routes the profile fine via attention to context tokens. Default
    # OFF; kept as a flag.
    inject_profiles_to_head: bool = False

    # Two-stage propensity TYPE head: the head reads `hidden[t]` PLUS clean
    # separated `type_emb[t]`, `zone_emb[t]`, `result_emb[t]` of pitch t,
    # combined via a small MLP. Mirrors the two-stage result head pattern —
    # the head no longer has to extract the previous pitch's factors from the
    # trunk's factored-sum residual stream; they're passed in directly.
    propensity_type_two_stage: bool = False

    # Per-pitch arsenal feature (ADR 009). When True, the 14-dim
    # arsenal+has-pitch sub-vector of the pitcher profile (7 usage rates + 7
    # binary "has thrown" flags, raw 0-1) is projected via Linear(14, d_model)
    # and ADDED to every pitch token's embedding — so the pitcher's pitch mix
    # is in the residual stream at every position, with no attention hop back
    # to the context token (the form the LSTM baseline gets it in).
    #
    # Default OFF so checkpoints predating ADR 009 (whose saved config has no
    # such key) reload without a state-dict mismatch. `train_pitchgpt.train()`
    # defaults it ON for new runs; new checkpoints save it as True.
    arsenal_per_pitch: bool = False
    n_arsenal_dims: int = 14  # 7 arsenal usage rates + 7 has_pitch flags

    # Situational two-stage propensity head (ADR 010). The propensity heads
    # (type/zone/velo/spin) predict pitch t+1, so they should condition on the
    # SITUATION pitch t+1 is thrown in — count[t+1], runners[t+1], outs[t+1] —
    # which is known at decision time. Without this, the trunk only has the
    # situation of pitch t (count[t], etc.) and must *derive* count[t+1] from
    # count[t] + result[t]; the LSTM baseline, being discriminative, gets the
    # current pitch's count as a direct feature for free. When True, a small
    # fusion MLP combines the trunk hidden with the clean embeddings of
    # (count, runners, outs)[t+1] before the propensity heads — same pattern as
    # the two-stage RESULT head conditioning on the (intervened) action.
    # Default OFF for checkpoint back-compat; `train()` defaults it ON. Not
    # meant to be combined with `propensity_type_two_stage`.
    propensity_situational: bool = False

    # Concat-then-project per-pitch factor embeddings (ADR 011, "fix #1").
    # Default: the 11 per-pitch factor embeddings are *summed* into one
    # d_model token, forcing the trunk to disentangle the sum. When True:
    # each factor's d_model embedding is down-projected to d_model//11 dims,
    # the 11 are concatenated, and a Linear(11*(d//11), d_model) mixes them —
    # giving each factor a dedicated input sub-space + a *learned* (not
    # forced-equal) mixing. Embedding tables stay d_model-dim, so the
    # weight-tied propensity heads and the two-stage result head are untouched.
    # Default OFF for checkpoint back-compat; `train()` flag controls new runs.
    concat_then_project: bool = False

    # FiLM-condition the transformer trunk on the player profile (ADR 012,
    # "fix #2"). Default: the profile reaches pitch tokens only by attending
    # to a prepended context token. When True: an MLP maps the (pitcher ++
    # batter) profile to per-layer (gamma, beta), and each transformer block's
    # input is modulated `gamma_l * x + beta_l` — so the profile conditions the
    # whole network (the transformer analogue of the LSTM's profile-in-h0).
    # Init: identity (gamma=1, beta=0) so it's a no-op at start. Default OFF
    # for checkpoint back-compat; `train()` flag controls new runs.
    profile_film: bool = False

    # --- Profile dims (per profile_cache schema) ---
    # v5 (2026-05-14, 14-zone migration): heatmap/grid dims drop with
    # N_IN_ZONE_CELLS 25→9. Pitcher: 230 → 118 (lost 7×16 dead heatmap slots).
    # Batter:  105 → 57  (lost 3×16 dead zone-grid slots).
    # v6 (2026-05-17, conditional-arsenal expansion): drop 12 entropy dims, add
    # 84 per-(type × count) + 14 per-(type × stand) + 14 movement. Pitcher:
    # 118 → 218. Batter schema unchanged.
    pitcher_profile_dim: int = 218
    batter_profile_dim: int = 57

    # --- Categorical confounder vocabularies ---
    n_p_throws: int = 3      # R, L, PAD/UNK
    n_stand: int = 3         # R, L, PAD/UNK (or 4 for switch)
    n_ballparks: int = 64    # ~30 MLB ballparks + buffer for relocated/COVID
    n_umpires: int = 256     # ~100 active umpires + buffer
    n_catchers: int = 384    # ~200 active catchers + buffer
    # ADR 003 Amendment 1 (2026-05-10): leverage replaced by raw state components.
    n_inning_buckets: int = 14    # innings 1..12, extras=13, PAD=0
    n_score_diff_buckets: int = 11  # signed, clipped to [-5, +5] → 11 values, no PAD
    n_inning_half: int = 3        # top, bot, PAD
    n_days_rest_buckets: int = 9  # 0..7+ days, plus PAD
    n_tto_buckets: int = 5        # 1st/2nd/3rd/4th+/PAD
    n_temp_buckets: int = 7       # cold/cool/mild/warm/hot/very-hot/missing
    n_roof: int = 3               # open/closed/missing

    # Per-pitch fatigue factor (ADR 003 Amendment 1)
    n_pitcher_fatigue_buckets: int = 12  # 0..9, 10..19, ..., 100+, PAD

    # --- Output head sizes ---
    n_result_logits: int = 7        # 7 result classes (no PAD at output)
    n_ab_outcome_classes: int = 7   # K, BB, 1B, 2B, 3B, HR, out

    # --- Training-time things, not architectural but useful to keep here ---
    label_smoothing_type: float = 0.05
    head_weights: dict = field(default_factory=lambda: {
        "type": 2.0,
        "zone": 2.0,
        "velo": 1.0,
        "spin_rate": 1.0,
        "spin_axis": 1.0,
        "result": 1.5,
        "ab_outcome": 1.0,
    })

    # Auxiliary spatial loss on the zone head (v5 14-zone investigation,
    # 2026-05-14). Default CE loss treats every wrong zone equally, which
    # creates a marginal-mode bias: fold-0 Tiny argmax-predicts z12 (lower-
    # right OOZ) 55% of the time vs 19% marginal. This aux term penalizes
    # the squared distance between ``E_p[zone_centroid]`` and the true zone
    # centroid in (plate_x, plate_z) feet, with centroids loaded from
    # ``data/preprocess_artifacts/v2/zone_centroids.npy`` (shape (13, 2)).
    # The categorical CE is unchanged so the propensity head remains a valid
    # probability distribution for the causal layer — this is an EMD-style
    # smoothing on top, not a replacement. Set 0 to disable (back-compat
    # default; checkpoints predating this knob load with 0.0).
    zone_spatial_weight: float = 0.0

    # Focal loss on the type head (Lin et al 2017). Replaces CE with
    # (1 - p_true)^gamma * CE, downweighting examples the model is already
    # confident about. Motivated by the empirical observation that fold-0
    # over-predicts FF by +7.6pp (39% pred vs 31% true) and under-predicts
    # CU by -3.1pp — focal loss should focus capacity on the hard cases.
    # gamma=0 disables (default = CE). gamma=2 is the canonical setting.
    type_focal_gamma: float = 0.0

    # Inverse-frequency class weighting on the type head. Multiplies the
    # per-sample focal loss by (1/freq_class)^class_weight_alpha. alpha=0
    # disables. alpha=0.5 is a moderate setting; alpha=1.0 is "fully balanced"
    # like sklearn's "balanced" mode. Requires ``type_class_freq`` to be set.
    type_class_weight_alpha: float = 0.0
    # Empirical pitch-type frequencies (length n_pitch_types). Computed once
    # from training corpus before training starts; None means "compute as needed".
    type_class_freq: list = None

    def __post_init__(self):
        # Sanity checks
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads "
                f"({self.n_heads})"
            )
        if self.d_ff is None:
            self.d_ff = 4 * self.d_model


# ---------- factory helpers ----------


def sanity_config() -> PitchGPTConfig:
    """v0 throwaway: small enough to verify the training loop works."""
    return PitchGPTConfig(
        n_layers=2,
        n_heads=2,
        d_model=128,
        d_ff=512,
        dropout=0.1,
    )


def tiny_config() -> PitchGPTConfig:
    """Default training target. ~6M params. Per pitchgpt-model skill."""
    return PitchGPTConfig(
        n_layers=4,
        n_heads=4,
        d_model=256,
        d_ff=1024,
        dropout=0.1,
    )


def small_config() -> PitchGPTConfig:
    """Ablation target if Tiny calibrates clean. ~25M params."""
    return PitchGPTConfig(
        n_layers=6,
        n_heads=8,
        d_model=512,
        d_ff=2048,
        dropout=0.1,
    )
