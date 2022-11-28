import re
from typing import Optional

import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map, FrozenLakeEnv
from gymnasium.vector.utils import spaces

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": ["SFFF", "FHFH", "FFFH", "HFFG"],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
    "12x12":
        ['SFHFFFFFFFHH',
         'FFFFFFFFFFFF',
         'FFFFFHFFFFHH',
         'FFHFFFFFFFFF',
         'FFFFFFFFFFHF',
         'HFHHHFHFFFFF',
         'FFFFFFFHFFHF',
         'FFHFFFFFHFFF',
         'FFHFFHFFFFFF',
         'FFHFFFHHFFFF',
         'HFFFFHHHFFFF',
         'HFFFFHFFFFHG'],
    "16x16":
        ['SHFFFFHFFHFFFFHF',
         'FFFFHFFFFFFFHFFF',
         'FFFFFFFFFFFFFHFF',
         'FHFFHFFHFFHHHFFH',
         'FFHFFFFHFHFFHHHF',
         'FHHFFFFFHFHHFHFF',
         'FFFFFFHFFFFFFFHF',
         'FFHFHHFFFFFFFFFF',
         'FFHFFFFFFFFFHFFF',
         'FFFFFFFFFFFFHHFF',
         'HFFHHHFFFHFFFFFF',
         'FHFHFFFFFFFFFFFF',
         'FHHFHFFFFFHHFFFF',
         'FFFFFFFFFFFFHFFF',
         'FFFFFHFFFFHFFFFH',
         'FFFFFHFFFFFFFHFG'],
}


class FrozenLakeEnvShapedMDPCompatible:
    """Class to convert Discrete Open AI Gym environemnts to MDPToolBox environments.

    You can find the list of available gym environments here: https://gym.openai.com/envs/#classic_control

    You'll have to look at the source code of the environments for available kwargs; as it is not well documented.
    """

    def __init__(self, openAI_env_name: str, **kwargs):
        """Create a new instance of the OpenAI_MDPToolbox class

        :param openAI_env_name: Valid name of an Open AI Gym env
        :type openAI_env_name: str
        :param render: whether to render the Open AI gym env
        :type rander: boolean
        """
        self.env_name = openAI_env_name

        self.env = FrozenLakeEnvShaped(**kwargs)
        self.env.reset()

        self.transitions = self.env.P
        self.actions = int(re.findall(r'\d+', str(self.env.action_space))[0])
        self.states = int(re.findall(r'\d+', str(self.env.observation_space))[0])
        self.P = np.zeros((self.actions, self.states, self.states))
        self.R = np.zeros((self.states, self.actions))
        self.convert_PR()

    def convert_PR(self):
        """Converts the transition probabilities provided by env.P to MDPToolbox-compatible P and R arrays
        """
        for state in range(self.states):
            for action in range(self.actions):
                for i in range(len(self.transitions[state][action])):
                    tran_prob = self.transitions[state][action][i][0]
                    state_ = self.transitions[state][action][i][1]
                    self.R[state][action] += tran_prob * self.transitions[state][action][i][2]
                    self.P[action, state, state_] += tran_prob



class FrozenLakeEnvShaped(FrozenLakeEnv):
    """
    Customized the initialization of the Frozen Lake Environment so that we can add reward shaping
    """

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        render_mode: Optional[str] = None,
        desc=None,
        map_name="4x4",
        is_slippery=True,
    ):
        if desc is None and map_name is None:
            desc = generate_random_map()
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc, dtype="c")
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        nA = 4
        nS = nrow * ncol

        self.initial_state_distrib = np.array(desc == b"S").astype("float64").ravel()
        self.initial_state_distrib /= self.initial_state_distrib.sum()

        self.P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row * ncol + col

        def inc(row, col, a):
            if a == LEFT:
                col = max(col - 1, 0)
            elif a == DOWN:
                row = min(row + 1, nrow - 1)
            elif a == RIGHT:
                col = min(col + 1, ncol - 1)
            elif a == UP:
                row = max(row - 1, 0)
            return (row, col)

        def update_probability_matrix(row, col, action):
            newrow, newcol = inc(row, col, action)
            newstate = to_s(newrow, newcol)
            newletter = desc[newrow, newcol]
            terminated = bytes(newletter) in b"GH"

            if newletter == b"G":
                reward = float(newletter == b"G")
            elif newletter == b"H":
                reward = float(-0.1)
            else:
                reward = 0

            return newstate, reward, terminated

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = self.P[s][a]
                    letter = desc[row, col]
                    if letter in b"GH":
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a - 1) % 4, a, (a + 1) % 4]:
                                li.append(
                                    (1.0 / 3.0, *update_probability_matrix(row, col, b))
                                )
                        else:
                            li.append((1.0, *update_probability_matrix(row, col, a)))

        self.observation_space = spaces.Discrete(nS)
        self.action_space = spaces.Discrete(nA)

        self.render_mode = render_mode

        # pygame utils
        self.window_size = (min(64 * ncol, 512), min(64 * nrow, 512))
        self.cell_size = (
            self.window_size[0] // self.ncol,
            self.window_size[1] // self.nrow,
        )
        self.window_surface = None
        self.clock = None
        self.hole_img = None
        self.cracked_hole_img = None
        self.ice_img = None
        self.elf_images = None
        self.goal_img = None
        self.start_img = None

