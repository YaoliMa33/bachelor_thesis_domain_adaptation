"""Training parameters and model selection for ST."""
from dataclasses import dataclass

DATASET = "ST"

METHODS = ("DANN", "DANNLC", "CDAN", "CDANLC")


@dataclass(frozen=True)
class Config:
    input_dim: int = 54
    classes: int = 3
    hidden_dim: int = 200
    domain_hidden: int = 32
    batch_size: int = 20
    epochs: int = 60
    grl_ramp_updates: int | None = 150
    steps_per_epoch: int = 5
    optimizer: str = 'Adam'
    learning_rate: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0001
    domain_weight: float = 0.9
    lc_weight: float = 1.2
    scheduler: str = 'constant'
    eta_min: float = 1e-06
    gamma: float = 3.0
    spectral_cap: float = 0.75
    epsilon: float = 1e-08
    # Class order: air, alcohol, acetone.
    source_sampling: str = 'stratified'
    source_class_counts: tuple[int, ...] = (4, 8, 8)
    # 完全随机采样（不固定类别配额）：取消下一行注释即可启用。
    # source_sampling: str = 'loader'
    target_sampling: str = 'loader'

