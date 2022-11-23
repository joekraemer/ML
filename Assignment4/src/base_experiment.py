import os
import logging
import time
from typing import Type

from hiive.mdptoolbox.mdp import MDP

from Assignment4.src.environments.environments import BaseEnvironment
from Assignment4.src.solvers.base_solver import BaseSolver

import pandas as pd


logger = logging.getLogger(__name__)


class BaseExperiment:
    def __init__(self, cfg, environment: Type[BaseEnvironment], solver: Type[BaseSolver], solver_name: str):
        self.cfg = cfg

        # Setup the environment and solver
        self.environment = environment(cfg=cfg)
        env = self.environment.build()
        self.solver = solver()
        self.solver.build(env, cfg=cfg[self.environment.Name][self.solver.Name])

        self.solver_name = solver_name
        self.environment_name = self.environment.Name

        out = '{}/{}/{}'.format(cfg.paths.results_data, self.environment_name, solver_name)
        if not os.path.exists(out):
            os.makedirs(out)
        self._out = '{}/{}'.format(out, '{}')
        return

    async def run(self) -> None:
        self.solver.run()

        logger.info('Logging experiment results')
        res_df = pd.DataFrame(self.solver.get_run_stats())
        res_df.to_csv(self._out.format('run_stats.csv'), index=False)

        res_df = pd.DataFrame(self.solver.get_final_policy())
        res_df.to_csv(self._out.format('final_policy.csv'), index=False)
        return

    def get_name(self):
        return self.environment.Name + ' ' + self.solver_name

