"""
Configuration for SpectraGC
"""

from dataclasses import dataclass


@dataclass
class MSGCConfig:

    hidden_dim: int = 64

    coarsening_ratio: float = 0.5

    learning_rate: float = 5e-3

    weight_decay: float = 5e-4

    epochs: int = 200

    seed: int = 42

    device: str = "cuda"
