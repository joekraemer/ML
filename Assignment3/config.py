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
class NNGSParams:
    layers: List[int] = field(default_factory=list)
    nodes: List[int] = field(default_factory=list)
    lr: List[int] = field(default_factory=list)


@dataclass
class DimRedAnalysis:
    dims: List[int] = field(default_factory=list)


@dataclass
class ClusterAnalysis:
    clusters: List[int] = field(default_factory=list)


@dataclass
class NNValues:
    # NN Params
    nodes: int
    lr: float
    alpha: float
    arch: List[int] = field(default_factory=list)


@dataclass
class ClusterValues(NNValues):
    clusters: int = 8


@dataclass
class DimRedValues(NNValues):
    dims: int = 8


@dataclass
class FeatureImportance(NNValues):
    features_to_keep: List[int] = field(default_factory=list)


@dataclass
class BestValues:
    # Best Values
    ica: DimRedValues
    pca: DimRedValues
    rp: DimRedValues
    fi: FeatureImportance
    gmm: ClusterValues
    kmm: ClusterValues


@dataclass
class Dataset:
    name: str
    paths: str


@dataclass
class A3Config:
    paths: Paths
    gs_params: NNGSParams
    dr_analysis: DimRedAnalysis
    cluster_analysis: ClusterAnalysis
    redwine: BestValues
    diabetic: BestValues

    verbose: bool
    threads: int
    seed: int
    aws: bool
    nn_early_stopping: bool
    learning_curves_n_training_sizes: int
    loss_curves_n_curves: int
    cv_splits: int

    dataset1: bool
    dataset2: bool

    benchmark: bool
    ica: bool
    pca: bool
    rp: bool
    fi: bool
    clustering: bool
    kmeans: bool
    gmm: bool

    run_basic: bool
    run_nn: bool
    run_additional_clustering: bool

    plot: bool
    plot_old_results: bool
    old_results_path: str
