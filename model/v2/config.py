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
    n_continuous: int = 4           # velo, spin, plate_x, plate_z
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
