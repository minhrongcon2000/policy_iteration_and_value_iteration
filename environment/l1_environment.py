from environment.base_environment import BaseEnvironment


class L1Environment(BaseEnvironment):
    def reward(self, state, action):
        return 1 / (1 + abs(self.end_pos[0] - state[0]) + abs(self.end_pos[1] - state[1]))