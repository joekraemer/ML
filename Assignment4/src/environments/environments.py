from abc import abstractmethod

from hiive.mdptoolbox import mdp, example
from gymnasium.envs.toy_text.frozen_lake import generate_random_map


class BaseEnvironment:
    def __init__(self, cfg):
        self.cfg = cfg
        self.Name = ''
        pass

    @abstractmethod
    def build(self):
        raise NotImplementedError


class ForestEnvironment(BaseEnvironment):
    def __init__(self, cfg, size: int = 10, p: float = 0.98):
        super().__init__(cfg)
        self.Size = size
        self.P = p
        self.Name = 'forest'

    def build(self, size: int = 10, p: float = 0.98):
        P, R = example.forest(S=size, p=p, is_sparse=True)
        return P, R


class FrozenLakeEnvironment(BaseEnvironment):
    def __init__(self, cfg, size: int = 10, p: float = 0.98):
        super().__init__(cfg)
        self.Size = size
        self.P = p
        self.Name = 'frozen_lake'

    def build(self, size: int = 10, p: float = 0.98):
        random_map = generate_random_map(size=size, p=p)
        P, R = example.openai("FrozenLake-v1", desc=random_map)
        return P, R

