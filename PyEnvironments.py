import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping
from IPython.display import display, clear_output
from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts

class CurveEnv(py_environment.PyEnvironment):
    
    def __init__(self, verbose: bool=False):
        self._low = -3
        self._high = 3

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.int32, minimum=self._low, maximum=self._high, name='observation')

        # I.e. 0 corresponds to "decrease", 1 to "increse"
        # self._action_to_direction = {
        #     0: -1,
        #     1: 1,
        # }
        self._state = 0
        self._episode_ended = False
        self._verbose = verbose

    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):
        self._episode_ended = False
        self._target_location = self._get_target()
        self._state = self._get_initial_state()

        # We will sample agent's location until 
        # it does not coincide with the target location
        while np.allclose(self._state, self._target_location): 
            self._state = self._get_initial_state()
        
        self._best_state = self._state

        # self.ax.clear()

        # if self.render_mode == "human":
        #     self._render()

        if self._verbose: print(f"[RESET] state: {self._state}, target: {self._target_location}")

        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()
        
        action = self._transform_action(action)
        # `_state` is a np.array([x, y]), so first we update state_x
        self._state[0] += action
        # We use `np.clip` to make sure we don't leave the boundaries
        self._state[0] = np.clip(self._state[0], self._low, self._high)
        self._state[1] = self._y(self._state[0])

        # A metric (y, throughput, latency, etc) is a reward itself
        reward = -self._state[1]

        # An episode is done if the agent has reached the target.
        self._episode_ended = np.allclose(self._state, self._target_location)
        
        # self.ax.clear()

        # if self.render_mode == "human":
        #     self._render_frame()

        if self._verbose: print(f"[STEP] state: {self._state}, target: {self._target_location}, reward: {reward}")

        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype=np.int32), reward)
        else:
            return ts.transition(
          np.array(self._state, dtype=np.int32), reward=0.0, discount=1.0)

    def _y(self, x):
        return 2*math.pow(x, 3) - 3*math.pow(x, 2) - 12 * x + 1
    
    def _transform_action(self, action):
        # Convert action from [0, 1] to [-1, 1]
        return 2 * action - 1
    
    def _get_target(self):
        x0 = [1.]
        minimizer_kwargs = {"method": "BFGS"}
        solution = basinhopping(self._y, x0, minimizer_kwargs=minimizer_kwargs,
                   niter=200)
        target_x = np.round(solution.x[0], 5)
        target_y = np.round(solution.fun, 5)
        return np.array([target_x, target_y], dtype=np.int32)
    
    def _get_initial_state(self):
        state_x = np.random.randint(self._low, self._high, size=None, dtype=int)
        state_y = self._y(state_x)
        return np.array([state_x, state_y], dtype=np.int32)
    
    def _get_reward(self, next_state, current_state):
        """
        The reward is the relative difference between
        a current agent value and the next one. The 
        normalization puts a large measurement range on the
        same scale.
        
        Han, Xue & Yu, Tingting. (2020). Automated Performance Tuning for Highly-Configurable Software Systems. 
        """
        return (next_state - current_state) / current_state

    def _render(self): 
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Agent Learning Curve Navigation')
         
        x_vals = np.linspace(self._low, self._high)
        y_vals = [self._y(x) for x in x_vals]
        
        self.ax.clear()
        self.ax.grid(True)
        self.ax.plot(x_vals, y_vals, color='black')
        self.ax.scatter(self._target_location[0], self._target_location[1], color='red', label="Target")
        self.ax.scatter(self._state[0], self._state[1], color='blue', label="Agent")
        
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_title('Agent Learning Curve Navigation')
        self.ax.legend()
        clear_output(wait=True)
        display(self.fig)