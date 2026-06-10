from dataclasses import dataclass


@dataclass
class V2Config:
    # Trunk
    n_layers: int = 4
    n_heads: int = 4
    d_model: int = 256
    d_ff: int = 1024
    dropout: float = 0.1
    init_std: float = 0.02

    # Input
    n_pitch_types: int = 8          # 7 types + PAD (input embedding only)
    n_type_classes: int = 7         # output softmax classes (no PAD)
    # Continuous dims, APPEND-ONLY order (older 4-dim checkpoints store their
    # own config and load unchanged): velo, spin, plate_x, plate_z,
    # spin_axis_sin, spin_axis_cos. v1c.1 added the spin axis so the cascade
    # stops receiving zeros for its spin_axis_sin/cos features.
    n_continuous: int = 6
    n_result_classes: int = 8       # 7 results + "none" for position 0
    n_count_states: int = 12
    n_runner_states: int = 8
    n_outs: int = 3
    max_positions: int = 15

    # adaLN conditioning
    pitcher_profile_dim: int = 223
    batter_profile_dim: int = 91
    adaln_hidden: int = 1024

    # GMM output
    gmm_components: int = 5
    gmm_logstd_floor: float = -3.0
    gmm_logstd_ceil: float = 2.0

    # Loss weights
    w_type: float = 2.0
    w_continuous: float = 1.0
    label_smoothing: float = 0.05
    type_focal_gamma: float = 0.0

    # Continuous normalization (z-score: (x - mean) / std)
    # Precomputed from training data (2017-2023, 4.74M pitches; spin-axis
    # constants computed 2026-06-10 over the same population)
    continuous_means: tuple = (88.38, 2254.70, 0.04, 2.24, -0.0687, -0.4601)
    continuous_stds: tuple = (6.03, 361.77, 0.85, 0.98, 0.6663, 0.5805)

    # Noise injection
    noise_p: float = 0.0
    noise_ramp_steps: int = 3000


def tiny_v2_config() -> V2Config:
    return V2Config()


def small_v2_config() -> V2Config:
    return V2Config(
        n_layers=6,
        n_heads=8,
        d_model=512,
        d_ff=2048,
        adaln_hidden=2048,
    )
