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
        self._env = self.environment.build()
        self.solver = solver()
        self.solver.build(self._env, cfg=cfg[self.environment.Name][self.solver.Name])

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
        self._save_run_stats()
        self._save_final_policy()
        return

    def _save_final_policy(self, name: str = 'final_policy.csv') -> None:
        res_df = pd.DataFrame(self.solver.get_final_policy())
        res_df.to_csv(self._out.format(name), index=False)
        return

    def _save_run_stats(self, name: str = 'run_stats.csv'):
        res_df = pd.DataFrame(self.solver.get_run_stats())
        res_df.to_csv(self._out.format(name), index=False)
        return

    def get_name(self):
        return self.environment.Name + ' ' + self.solver_name


class ExploreExploitExperiment(BaseExperiment):
    """Experiment to investigate the exploration exploitation tradeoff"""

    def __init__(self, cfg, environment: Type[BaseEnvironment], solver: Type[BaseSolver], solver_name: str):
        super().__init__(cfg, environment, solver, solver_name)

        self.epsilon_decay_values = cfg[self.environment.Name].QLearning.epsilon_decay_values
        self.SolverClass = solver
        pass

    async def run(self) -> None:
        reward_df = pd.DataFrame()
        epsilon_df = pd.DataFrame()

        for eps in self.epsilon_decay_values:
            solver = self.SolverClass()
            temp_cfg = self.cfg[self.environment.Name][self.solver.Name].copy()
            temp_cfg['epsilon_decay'] = eps
            solver.build(self._env, cfg=temp_cfg)
            solver.run()
            stats = pd.DataFrame(solver.get_run_stats())
            stats = stats.set_index('Iteration')
            reward_df[eps] = stats['Reward']
            epsilon_df[eps] = stats['Epsilon']

        reward_df.to_csv(self._out.format('explore_exploit_reward.csv'), index=True)
        epsilon_df.to_csv(self._out.format('explore_exploit_epsilon.csv'), index=True)
        return


class StateSpaceSizeExperiment(BaseExperiment):
    """Experiment to investigate the exploration exploitation tradeoff"""

    def __init__(self, cfg, environment: Type[BaseEnvironment], solver: Type[BaseSolver], solver_name: str):
        super().__init__(cfg, environment, solver, solver_name)

        self.epsilon_decay_values = cfg[self.environment.Name].QLearning.epsilon_decay_values

        self.SolverClass = solver
        pass

    async def run(self) -> None:

        df = pd.DataFrame()

        for eps in self.epsilon_decay_values:
            solver = self.SolverClass()
            base_cfg = self.cfg[self.environment.Name][self.solver.Name]
            cfg = base_cfg ** {'epsilon_decay': eps}
            solver.build(self.environment, cfg=cfg)
            solver.run()
            stats = solver.get_run_stats()
            df[eps] = stats['Reward']
        return
