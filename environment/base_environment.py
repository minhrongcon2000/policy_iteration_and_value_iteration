import numpy as np
from enum import Enum
import seaborn as sns
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Ceil(Enum):
    OBSTACLES = 0.5
    AGENT = 0.1
    TERMINAL = 1

class Action(Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"

class BaseEnvironment(ABC):
    def __init__(self, size, exit_p=0.3, traffic_p=0.75):
        self.__size = size
        self.__exit_p = exit_p
        self.__traffic_p = traffic_p
        self.__start_pos = (0, 0)
        self.__end_pos = (self.__size-1, self.__size-1)
        self.__create_env()
    
    def __create_env(self):
        self.__env = np.zeros((self.__size, self.__size))
        self.__env[self.__start_pos[0], self.__start_pos[1]] = Ceil.TERMINAL.value
        self.__env[-1, -1] = Ceil.TERMINAL.value
        for i in range(self.__size):
            for j in range(self.__size):
                if (i, j) not in {(0, 0), (self.__size - 1, self.__size - 1)}:
                    if np.random.binomial(1, self.__traffic_p) > 0:
                        self.__env[i, j] = Ceil.OBSTACLES.value
    
    def next_position(self, state, action):
        if action == Action.UP.value:
            return (state[0] - 1, state[1])
        
        if action == Action.LEFT.value:
            return (state[0], state[1] - 1)
        
        if action == Action.RIGHT.value:
            return (state[0], state[1] + 1)
        
        return (state[0] + 1, state[1])
    
    def calculate_transitive_prob(self, next_state, current_state, action):
        if self.__env[current_state[0], current_state[1]] == Ceil.OBSTACLES.value:
            if next_state == current_state:
                return 1 - self.__exit_p
            
            if self.next_position(current_state, action) == next_state:
                return self.__exit_p
        
        if self.next_position(current_state, action) == next_state:
            return 1
        
        return 0
    
    def expected_value(self, current_state, v, action):
        n, n = v.shape
        s = 0
        for i in range(n):
            for j in range(n):
                s += self.calculate_transitive_prob((i, j), current_state, action) * v[i, j]
        return s
    
    def q(self, state, action, discounted_factor, v):
        return self.reward(state, action) + discounted_factor * self.expected_value(state, v, action)
    
    def perform_policy(self, policy):
        agent_path = self.env_agent[:]
        current_pos = self.__start_pos
        while current_pos != self.__end_pos:
            current_pos = self.next_position(state=current_pos, action=policy[current_pos[0], current_pos[1]])
            agent_path[current_pos[0], current_pos[1]] = Ceil.AGENT.value
        return agent_path
    
    @abstractmethod
    def reward(self, state, action):
        pass
        
    @property
    def env_array(self):
        return self.__env
    
    @property
    def size(self):
        return self.__size
    
    @property
    def traffic_p(self):
        return self.__traffic_p
    
    @property
    def start_pos(self):
        return self.__start_pos
    
    @property
    def end_pos(self):
        return self.__end_pos
    
    @property
    def env_agent(self):
        env = np.zeros((self.__size, self.__size))
        env[self.__start_pos[0], self.__start_pos[1]] = Ceil.AGENT.value
        return env