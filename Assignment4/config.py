from dataclasses import dataclass, field

from typing import List


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

    verbose: bool
    threads: int
    seed: int
    aws: bool

    forest: bool
    frozen_lake: bool

    plot: bool
    plot_old_results: bool
    old_results_path: str
