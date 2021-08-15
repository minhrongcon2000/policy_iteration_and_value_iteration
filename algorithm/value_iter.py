import numpy as np 
from environment.base_environment import Action


class GridValueIteration:
    def __init__(self, env, discount_factor):
        self.__env = env
        self.__discount_factor = discount_factor
        self.__possible_actions = [Action.UP.value, Action.DOWN.value, Action.LEFT.value, Action.RIGHT.value]
        self.__optimal_policy = np.empty((self.__env.size, self.__env.size), dtype="<U5")
        self.__v = np.zeros((self.__env.size, self.__env.size))
        
    def __find_best_action(self, state):
        max_q = None
        best_action = None
        for action in self.__possible_actions:
            action_value = self.__env.q(state, action, self.__discount_factor, self.__v)
            if max_q is None or action_value > max_q:
                max_q = action_value
                best_action = action
        return max_q, best_action
    
    def solve_env(self, threshhold=1e-6):
        done = False
        new_v = np.zeros((self.__env.size, self.__env.size))
        while not done:
            for i in range(self.__env.size):
                for j in range(self.__env.size):
                    max_q, best_action = self.__find_best_action((i, j))
                    new_v[i, j] = max_q
                    self.__optimal_policy[i, j] = best_action
            if np.sum(np.abs(new_v - self.__v)) <= threshhold:
                done = True
            else:
                self.__v = new_v
                new_v = np.zeros((self.__env.size, self.__env.size))
    @property
    def optimal_policy(self):
        return self.__optimal_policy
    
    @property
    def value_mat(self):
        return self.__v