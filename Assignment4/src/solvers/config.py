from dataclasses import dataclass


@dataclass
class SolverConfig:
    gamma: float
    max_iter: int


@dataclass
class VIConfig(SolverConfig):
    epsilon: float


@dataclass
class QLearningConfig(SolverConfig):
    alpha: float
    alpha_decay: float
    epsilon_decay: float
    epsilon_min: float


@dataclass
class AllSolverConfig:
    VI: VIConfig
    PI: SolverConfig
    QLearning: QLearningConfig

