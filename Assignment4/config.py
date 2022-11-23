from dataclasses import dataclass, field

from typing import List

from Assignment4.src.solvers.config import AllSolverConfig


@dataclass
class Paths:
    root: str
    data: str
    wine: str
    diabetic: str
    results_data: str
    plots: str


@dataclass
class GSParams:
    layers: List[int] = field(default_factory=list)
    nodes: List[int] = field(default_factory=list)
    lr: List[int] = field(default_factory=list)


@dataclass
class A4Config:
    paths: Paths
    gs_params: GSParams

    forest: AllSolverConfig
    frozen_lake: AllSolverConfig

    verbose: bool
    threads: int
    seed: int
    aws: bool

    forest_enable: bool
    frozen_lake_enable: bool

    plot: bool
    plot_old_results: bool
    old_results_path: str
