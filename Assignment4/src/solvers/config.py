from dataclasses import dataclass, field
from typing import List


@dataclass
class PIConfig:
    gamma: float
    convergence_value: float


@dataclass
class VIConfig:
    gamma: float
    epsilon: float


@dataclass
class QLearningConfig:
    gamma: float
    alpha: float
    alpha_decay: float
    epsilon_decay: float
    epsilon_min: float
    convergence_value: float
    max_iterations: int
    epsilon_decay_values: List[int] = field(default_factory=list)


@dataclass
class AllSolverConfig:
    VI: VIConfig
    PI: PIConfig
    QLearning: QLearningConfig

