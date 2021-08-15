from environment.base_environment import BaseEnvironment
from environment.base_environment import Action
import numpy as np


class GridPolicyIteration:
    def __init__(self, env: BaseEnvironment, discounted_factor):
        self.__discounted_factor = discounted_factor
        self.__possible_actions = [Action.UP.value, Action.DOWN.value, Action.LEFT.value, Action.RIGHT.value]
        self.__env = env
        self.__optimal_policy = self.__initialize_policy(self.__env.size)
        
    def __initialize_policy(self, n):
        policies = np.empty((n, n), dtype="<U5")
        policies[0, 0] = np.random.choice([Action.RIGHT.value, Action.DOWN.value])
        policies[0,-1] = np.random.choice([Action.LEFT.value, Action.DOWN.value])
        policies[-1,0] = np.random.choice([Action.UP.value, Action.RIGHT.value])
        policies[-1,-1] = np.random.choice([Action.UP.value, Action.LEFT.value])
        policies[0, 1:-1] = np.random.choice([Action.DOWN.value, Action.LEFT.value, Action.RIGHT.value], size=n-2)
        policies[-1, 1:-1] = np.random.choice([Action.DOWN.value, Action.LEFT.value, Action.RIGHT.value], size=n-2)
        policies[1:-1, 0] = np.random.choice([Action.UP.value, Action.DOWN.value, Action.RIGHT.value], size=n-2)
        policies[1:-1, -1] = np.random.choice([Action.UP.value, Action.DOWN.value, Action.LEFT.value], size=n-2)
        policies[1:-1, 1:-1] = np.random.choice([Action.UP.value, Action.DOWN.value, Action.LEFT.value, Action.RIGHT.value], size=(n-2, n-2))
        return policies
        
    def __policy_evaluation(self, policy, env: BaseEnvironment, threshhold=1e-9):
        n = env.size
        v = np.zeros((n, n))
        new_v = np.zeros((n, n))
        done = False
        while not done:
            # calculate discounted sum of reward
            for i in range(n):
                for j in range(n):
                    new_v[i, j] = env.q(state=(i, j), 
                                        action=policy[i, j], 
                                        discounted_factor=self.__discounted_factor, 
                                        v=v)
            if np.abs(v - new_v).sum() <= threshhold:
                done = True
            else:
                v = new_v
                new_v = np.zeros((n, n))
        return v
    
    def __policy_improvement(self, policy, env: BaseEnvironment):
        n = env.size
        new_policy = np.empty_like(policy)
        for i in range(n):
            for j in range(n):
                max_q = None
                best_action = None
                for action in self.__possible_actions:
                    action_value = env.q(state=(i, j), 
                                         action=action, 
                                         discounted_factor=self.__discounted_factor, 
                                         v=self.__v)
                    if max_q is None or max_q < action_value:
                        max_q = action_value
                        best_action = action
                new_policy[i, j] = best_action
        return new_policy
    
    def solve_env(self, threshhold=1e-9):
        done = False
        i = 0
        while not done:
            print("Iteration {}".format(i+1))
            print("Policy Eval...")
            self.__v = self.__policy_evaluation(policy=self.__optimal_policy,
                                                env=self.__env,
                                                threshhold=threshhold)
            print("Policy improve...")
            new_policy = self.__policy_improvement(policy=self.__optimal_policy,
                                                  env=self.__env)
            if (new_policy == self.__optimal_policy).astype(int).sum() == self.__env.size ** 2:
                done = True
            else:
                self.__optimal_policy = new_policy
            i += 1
    @property
    def discounted_factor(self):
        return self.__discounted_factor
    
    @property
    def optimal_policy(self):
        return self.__optimal_policy
    
    @property
    def value_mat(self):
        return self.__v