"""Training parameters and model selection for UCI."""
from dataclasses import dataclass

DATASET = "UCI"

METHODS = ("DANN", "DANNLC", "CDAN", "CDANLC")


@dataclass(frozen=True)
class Config:
    input_dim: int = 128
    classes: int = 6
    hidden_dim: int = 200
    domain_hidden: int = 64
    batch_size: int = 16
    epochs: int = 45
    grl_ramp_updates: int | None = None
    steps_per_epoch: int = 10
    optimizer: str = 'SGD'
    learning_rate: float = 0.03
    momentum: float = 0.9
    weight_decay: float = 0.0
    domain_weight: float = 0.8
    lc_weight: float = 0.2
    scheduler: str = 'cosine'
    eta_min: float = 1e-06
    gamma: float = 10.0
    spectral_cap: float = 1.0
    epsilon: float = 1e-08
    source_sampling: str = 'stratified'
    target_sampling: str = 'continuous'

