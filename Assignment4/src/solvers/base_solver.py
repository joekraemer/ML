from abc import abstractmethod, ABC
from typing import Dict, List

from hiive.mdptoolbox import mdp

from Assignment4.src.environments.environments import FakeEnv
from Assignment4.src.solvers.QLearner import QLearner


class BaseSolver(ABC):
    def __init__(self):
        self.Name = ''
        self._solver = None
        self.cfg = None
        pass

    def run(self):
        self._solver.run()
        return

    @abstractmethod
    def build(self, env, cfg):
        return

    def get_run_stats(self) -> List[Dict]:
        return self._solver.run_stats

    def get_final_policy(self) -> List:
        temp = list(self._solver.policy)
        return temp


class StaticSolver(BaseSolver):
    """Solvers that just use P, V to solve a problem"""
    def __init__(self):
        super().__init__()

    def build(self, env, cfg):
        self.cfg = cfg
        self.desc = env.desc if type(env) is FakeEnv else env.env.desc
        self._build(env.P, env.R, cfg)
        return

    @abstractmethod
    def _build(self, P, R, cfg):
        pass


class QLearningSolver(BaseSolver):
    def __init__(self):
        super().__init__()
        self.Name = 'QLearning'

    def build(self, env, cfg):
        self.cfg = cfg

        if type(env) is FakeEnv:
            self.desc = env.desc
            self._solver = mdp.QLearning(env.P, env.R,
                                         gamma=cfg.gamma,
                                         epsilon_decay=cfg.epsilon_decay,
                                         epsilon_min=cfg.epsilon_min,
                                         alpha=cfg.alpha,
                                         alpha_decay=cfg.alpha_decay,
                                         convergence_value=cfg.convergence_value,
                                         n_iter=cfg.max_iterations)

        else:
            self.desc = env.env.desc
            self._solver = QLearner(env, cfg)
        return


class VISolver(StaticSolver):
    def __init__(self):
        super().__init__()
        self.Name = 'VI'
        return

    def _build(self, P, R, cfg):
        self._solver = mdp.ValueIteration(P, R, gamma=cfg.gamma, epsilon=cfg.epsilon)
        return


class PISolver(StaticSolver):
    def __init__(self):
        super().__init__()
        self.Name = 'PI'
        pass

    def _build(self, P, R, cfg):
        self._solver = mdp.PolicyIteration(P, R, gamma=cfg.gamma, convergence_value=cfg.convergence_value)
        return



