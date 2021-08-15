import numpy as np
from environment.base_environment import BaseEnvironment


class L2Environment(BaseEnvironment):
    def reward(self, state, action):
        return 1 / (1 + np.sqrt((self.end_pos[0] - state[0]) ** 2 + (self.end_pos[1] - state[1]) ** 2))