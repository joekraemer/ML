import os
from typing import Type

from hiive.mdptoolbox.mdp import MDP

from Assignment4.src.environments.environments import BaseEnvironment
import pandas as pd


class BaseExperiment:
    def __init__(self, cfg, environment: Type[BaseEnvironment], solver: Type[MDP], solver_name: str):
        self.cfg = cfg

        # Setup the environment and solver
        self.environment = environment(cfg=cfg)
        P, R = self.environment.build()
        self.solver = solver(P, R, 0.9)

        self.solver_name = solver_name
        self.environment_name = self.environment.Name

        out = '{}/{}/{}'.format(cfg.paths.results_data, self.environment_name, solver_name)
        if not os.path.exists(out):
            os.makedirs(out)
        self._out = '{}/{}'.format(out, '{}')
        return

    async def run(self) -> None:
        self.solver.run()

        print('Logging experiment results')
        res_df = pd.DataFrame(self.solver.run_stats)
        res_df.to_csv(self._out.format('run_stats.csv'), index=False)
        return

    def get_name(self):
        return self.environment.Name + ' ' + self.solver_name

