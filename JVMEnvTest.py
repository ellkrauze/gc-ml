import numpy as np
import gym
import subprocess
import re
import os
import math

import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

low = -100
high = 100
# low = 0
# high = 314
# low = -100 * math.pi
# high = 150 * math.pi

import os
import re
import sys
import gym
import math
import random
import requests
import json
import subprocess
import logging
import pygame
import numpy as np
from typing import List
from gym.spaces import Discrete, Box, Dict, MultiDiscrete
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy.optimize import basinhopping
from IPython.display import display, clear_output


class JVMEnvTest(gym.Env):

    metadata = {"render_modes": ["human"] }
    
    def __init__(self, render_mode=None):
        self.window_size = 512  # The size of the PyGame window
        
        # Actions we can take: down, up
        self.action_space = Discrete(2) # {0, 1}

        # I.e. 0 corresponds to "decrease", 1 to "increse"
        self._action_to_direction = {
            0: -1,
            1: 1,
        }

        # self.observation_space = Box(low=-np.inf, high=np.inf, dtype=np.float32, shape=(2,))
        self.observation_space = Box(low=-np.inf, high=np.inf, dtype=np.int, shape=(2,))

        # self.low = 0.0
        # self.high = self.coef * math.pi

        # self.low = -100
        # self.high = 100
        self.low = -3
        self.high = 3
        self.coef_low = 2
        self.coef_high = 6
        # target_x = 0

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Agent Learning Curve Navigation')
        # self.ax.set_xlim(self.low, self.high)
        # self.ax.set_ylim(self.y(self.low), self.y(self.high))

    def y(self, x):
        # return - (2/self.coef) * (x**2) 
        # return np.sin(np.divide(x, self.coef)) * self.coef * 2
        # return math.pow(x, 4) + self.coef * math.pow(x, 3) - self.coef * 2 * x
        return 2*math.pow(x, 3) - 3*math.pow(x, 2) - 12 * x + 1
        # return 4 * math.pow(x, 4) + 10*math.pow(x, 3) - 3 *math.pow(x, 2) - 12*x
    
    # def y_prime(self, x):
    #     return 2 * np.cos(np.divide(x, self.coef))
    
    def get_reward(self, next_state, current_state):
        """
        The reward is the relative difference between
        a current agent value and the next one. The 
        normalization puts a large measurement range on the
        same scale.
        
        Han, Xue & Yu, Tingting. (2020). Automated Performance Tuning for Highly-Configurable Software Systems. 
        """
        return (next_state - current_state) / current_state

    def step(self, action):
        # print('STEP')
        action = self._action_to_direction[action]
        # x = x + action ({-1, 1})
        self._agent_location[0] += action
        # We use `np.clip` to make sure we don't leave the boundaries
        self._agent_location[0] = np.clip(self._agent_location[0], self.low, self.high)

        # Multiply by (-1) in the task of minimization.
        # The agent is penalised with a reward of -1 for each timestep.
        # reward = -self.get_reward( self.y(self._agent_location[0]) , self._agent_location[1] )
        
        # y = y + y(x+action)
        self._agent_location[1] = self.y(self._agent_location[0])

        # reward = -self.get_reward( self.y(self._agent_location[0]) , self._agent_location[1] )
        # if self._agent_location[1] <= self._best_agent_location:
        #     self._best_agent_location = self._agent_location[1]
        #     reward = 1
        # else:
        #     reward = -1

        reward = -self._agent_location[1]

        # An episode is done if the agent has reached the target.
        done = np.allclose(self._agent_location, self._target_location)
        # if done: print("DONE") 
        # if not done: reward = -1
        observation = self._get_obs()

        info = {}
        # info = self._get_info()
        # print(self._get_info())

        self.ax.clear()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, info

    def reset(self, seed=None, options=None):
        # print("RESET")
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        
        self.coef = 2
        # self.coef = self.np_random.integers(self.coef_low, self.coef_high, size=None, dtype=int)
        
        # agent_x = self.np_random.uniform(self.low, self.high, size=None)
        agent_x = self.np_random.integers(self.low, self.high, size=None, dtype=int)
        
        # Find global minimum
        x0 = [1.]
        minimizer_kwargs = {"method": "BFGS"}
        target_x = basinhopping(self.y, x0, minimizer_kwargs=minimizer_kwargs,
                   niter=200)
        target_x = target_x.x[0]
        

        # We will sample agent's location until 
        # it does not coincide with the target location
        while agent_x == target_x:
            # agent_x = self.np_random.uniform(self.low, self.high, size=None)
            agent_x = self.np_random.integers(self.low, self.high, size=None, dtype=int)

        self._target_location = np.array([target_x, self.y(target_x)]) # {x, y}
        self._agent_location = np.array([agent_x, self.y(agent_x)]) # {x, y}

        self._best_agent_location = self._agent_location[1]
            
        # self._target_location = target_x, self.y(target_x) # {x, y}
        # self._agent_location = agent_x, self.y(agent_x) # {x, y}

        observation = self._get_obs()
        info = self._get_info()

        # print(info)
        
        self.ax.clear()

        if self.render_mode == "human":
            self._render_frame()

        return observation

    def render(self, mode):
        x_vals = np.linspace(self.low, self.high)
        y_vals = [self.y(x) for x in x_vals]
        
        self.ax.clear()
        self.ax.grid(True)
        self.ax.plot(x_vals, y_vals, color='black')
        self.ax.scatter(self._target_location[0], self._target_location[1], color='red', label="Target")
        self.ax.scatter(self._agent_location[0], self._agent_location[1], color='blue', label="Agent")
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Agent Learning Curve Navigation')
        self.ax.legend()
        clear_output(wait=True)
        display(self.fig)

    def close(self):
        pass
    
    def _get_obs(self):
        # return {"agent": self._agent_location, "target": self._target_location}
        return self._agent_location
    
    def _get_info(self):
        return {"agent": self._agent_location, "target": self._target_location}
