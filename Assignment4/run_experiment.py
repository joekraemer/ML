import asyncio
import logging
from datetime import datetime
from typing import List, Dict

import hydra
from hydra.core.config_store import ConfigStore

from Assignment4.src.environments.environments import ForestEnvironment, FrozenLakeEnvironment
from config import A4Config
from src import plotting
from src.base_experiment import ExperimentDetails, BaseExperiment


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def await_all_tasks(tasks):
    await asyncio.gather(*tasks)


async def run_experiment(experiment_details: List[ExperimentDetails], experiment: BaseExperiment,
                         cfg: A4Config, timings: Dict) -> None:
    """

    :param experiment_details:
    :param timings: dictionary of run times
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
cs.store(name="a4_config", node=A4Config)


@hydra.main(config_path="configs", config_name="config")
def main(cfg: A4Config) -> None:
    logger.info("Seed" + str(cfg.seed))
    logger.info("----------")

    environments = []
    if cfg.forest:
        environments.append(ForestEnvironment)
    if cfg.frozen_lake:
        environments.append(FrozenLakeEnvironment)

    tasks = []
    timings = {}

    logger.info("Running experiments")

    if cfg.benchmark:
        tasks.append(run_experiment(experiment_details, BenchmarkExperiment, cfg, timings))

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
