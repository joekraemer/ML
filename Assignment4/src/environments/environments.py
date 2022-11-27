from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple

from hiive.mdptoolbox import mdp, example, openai
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

from Assignment4.src.environments.FrozenLake import FrozenLakeEnvShaped, FrozenLakeEnvShapedMDPCompatible


class FakeEnv:
    """ So that ForestEnvironment can return an object with the same properties as the gymnasium environments"""
    def __init__(self, p, r, env=None):
        self.P = p
        self.R = r
        self.env = env
        self.desc = None


class BaseEnvironment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.Name = ''
        pass

    @abstractmethod
    def build(self):
        raise NotImplementedError


class ForestEnvironment(BaseEnvironment):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.Name = 'forest'

    def build(self, size: int = 20, p: float = 0.05) -> Tuple:
        P, R = example.forest(S=size, p=p, is_sparse=True)
        return FakeEnv(P, R)  # P is the transition matrix and R is the reward matrix


class FrozenLakeEnvironment(BaseEnvironment):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.Name = 'frozen_lake'

    def build(self, size: int = 5, p: float = 0.8) -> Tuple:
        #random_map = generate_random_map(size=size, p=p)
        env = FrozenLakeEnvShapedMDPCompatible("FrozenLake-v1", map_name='8x8', is_slippery=True)
        return env

    def base_build(self, size: int = 10, p: float = 0.8) -> Tuple:
        random_map = generate_random_map(size=size, p=p)
        env = openai.OpenAI_MDPToolbox("FrozenLake-v1", desc=random_map, is_slippery=True)

        # Add reward shaping to the environment
        return env

        #P, R = example.openai("FrozenLake-v1", desc=random_map, is_slippery=True)
        #return P, R  # P is the transition matrix and R is the reward matrix

