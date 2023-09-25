import asyncio
import logging
from datetime import datetime
from typing import List

import hydra
from hydra.core.config_store import ConfigStore

from config import A3Config
from data import loader
from experiments import plotting
from experiments.FI import FIExperiment
from experiments.ICA import ICAExperiment
from experiments.PCA import PCAExperiment
from experiments.RP import RPExperiment
from experiments.base import ExperimentDetails, BaseExperiment
from experiments.benchmark import BenchmarkExperiment
from experiments.clustering import GeneralClusteringExperiment, KMeansExperiment, GMMExperiment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def await_all_tasks(tasks):
    await asyncio.gather(*tasks)


async def run_experiment(experiment_details: List[ExperimentDetails], experiment: BaseExperiment,
                         cfg: A3Config, timings) -> None:
    """

    :param experiment_details:
    :param timings:
    :param cfg:
    :param experiment:
    """
    t = datetime.now()
    for details in experiment_details:
        exp = experiment(details, cfg=cfg)
        logger.info(" Details: {}".format(details))

        if cfg.run_basic:
            logger.info("Running {} experiment: {}".format(exp.experiment_name, details.ds_name))
            await exp.perform()

        if cfg.run_nn:
            logger.info("Running Neural Network analysis")
            # await exp.perform_nn_grid_search()
            await exp.perform_nn_performance_analysis()

        if cfg.run_additional_clustering:
            logger.info("Running additional clustering on top of DR")
            await exp.perform_cluster()

    t_d = datetime.now() - t
    timings[exp.experiment_name] = t_d.seconds
    return


cs = ConfigStore.instance()
cs.store(name="mnist_config", node=A3Config)


@hydra.main(config_path="configs", config_name="config")
def main(cfg: A3Config) -> None:
    logger.info("Seed" + str(cfg.seed))
    logger.info("Loading data")
    logger.info("----------")

    dataloaders = []
    if cfg.dataset1:
        dataloaders.append(loader.WineData(path=cfg.paths.wine, verbose=cfg.verbose, seed=cfg.seed))
    if cfg.dataset2:
        dataloaders.append(loader.DiabeticData(path=cfg.paths.diabetic, verbose=cfg.verbose, seed=cfg.seed))

    experiment_details = []

    for dl in dataloaders:
        dl.load_and_process()
        dl.build_train_test_split()
        dl.scale_standard()
        experiment_details.append(ExperimentDetails(dl, dl.name))

    tasks = []
    timings = {}

    if cfg.verbose:
        logger.info("----------")

    logger.info("Running experiments")

    # TODO Probably a better way to do this with a mapping or something
    if cfg.benchmark:
        tasks.append(run_experiment(experiment_details, BenchmarkExperiment, cfg, timings))

    if cfg.ica:
        tasks.append(run_experiment(experiment_details, ICAExperiment, cfg, timings))

    if cfg.pca:
        tasks.append(run_experiment(experiment_details, PCAExperiment, cfg, timings))

    if cfg.rp:
        tasks.append(run_experiment(experiment_details, RPExperiment, cfg, timings))

    if cfg.fi:
        tasks.append(run_experiment(experiment_details, FIExperiment, cfg, timings))

    if cfg.clustering:
        tasks.append(run_experiment(experiment_details, GeneralClusteringExperiment, cfg, timings))

    if cfg.kmeans:
        tasks.append(run_experiment(experiment_details, KMeansExperiment, cfg, timings))

    if cfg.gmm:
        tasks.append(run_experiment(experiment_details, GMMExperiment, cfg, timings))

    # Run experiment tasks
    asyncio.run(await_all_tasks(tasks))

    logger.info("Timings: {}".format(timings))

    if cfg.plot:
        if cfg.verbose:
            logger.info("----------")

        logger.info("Plotting results")
        plotting.plot_results(cfg)


if __name__ == '__main__':
    main()
