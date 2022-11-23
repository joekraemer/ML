import asyncio
import logging

import hydra
from hydra.core.config_store import ConfigStore

from Assignment4.src.base_experiment import BaseExperiment
from Assignment4.src.environments.environments import ForestEnvironment, FrozenLakeEnvironment
from Assignment4.src.solvers.base_solver import VISolver, PISolver, QLearningSolver

from config import A4Config
from Assignment4 import plotting

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def await_all_tasks(tasks):
    await asyncio.gather(*tasks)


async def run_experiment(exp: BaseExperiment) -> None:
    """

    :param experiment_details:
    :param timings: dictionary of run times
    :param cfg:
    :param experiment:
    """
    logger.info("Running {} environment with {} solver".format(exp.environment_name, exp.solver_name))
    await exp.run()

    return


cs = ConfigStore.instance()
cs.store(name="a4_config", node=A4Config)


@hydra.main(config_path="configs", config_name="config")
def main(cfg: A4Config) -> None:
    logger.info("Seed" + str(cfg.seed))
    logger.info("----------")

    environments = []
    if cfg.forest_enable:
        environments.append(ForestEnvironment)
    if cfg.frozen_lake_enable:
        environments.append(FrozenLakeEnvironment)

    tasks = []

    logger.info("Running experiments")

    # could be a cfg parameter, but not now
    for env in environments:
        exp = BaseExperiment(environment=env, solver=VISolver, solver_name='VI', cfg=cfg)
        tasks.append(run_experiment(exp))

        exp = BaseExperiment(environment=env, solver=PISolver, solver_name='PI', cfg=cfg)
        tasks.append(run_experiment(exp))

        exp = BaseExperiment(environment=env, solver=QLearningSolver, solver_name='QLearning', cfg=cfg)
        tasks.append(run_experiment(exp))

    # Run experiment tasks
    asyncio.run(await_all_tasks(tasks))

    if cfg.plot:
        if cfg.verbose:
            logger.info("----------")

        logger.info("Plotting results")
        plotting.plot_results(cfg)


if __name__ == '__main__':
    main()
