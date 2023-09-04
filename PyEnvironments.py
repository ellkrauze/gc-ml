import math
import os
import re
import sys
import gym
import random
import requests
import json
import subprocess
import logging
from typing import List
from gym.spaces import Discrete, Box
import numpy as np
import matplotlib.pyplot as plt
# from scipy.optimize import basinhopping
import scipy.optimize as optimize
from IPython.display import display, clear_output
from tf_agents.specs import array_spec
from tf_agents.environments import py_environment
from tf_agents.trajectories import time_step as ts

logging.basicConfig(format='%(asctime)s-ENV-%(levelname)s-%(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')

class CurveEnv(py_environment.PyEnvironment):
    
    def __init__(self, verbose: bool=False):
        self._low = -3
        self._high = 3

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(2,), dtype=np.int32, minimum=self._low, maximum=self._high, name='observation')

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
        solution = optimize.basinhopping(self._y, x0, minimizer_kwargs=minimizer_kwargs,
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

class CurveMultipleEnv(py_environment.PyEnvironment):
    
    def __init__(self, verbose: bool=False):
        self._num_variables = 2

        self._flags_min = {
            "x": -2,
            "y": -10,
        }

        self._flags_max = {
            "x": 2,
            "y": 10,
        }

        self._action_mapping = {
            0: self._decrease_MaxHeapSize,
            1: self._increase_MaxHeapSize,
            2: self._decrease_InitialHeapSize,
            3: self._increase_InitialHeapSize,
        }

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2*self._num_variables-1, name='action') # {0, 1, 2, 3}
        
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._num_variables,), 
            dtype=np.int32, 
            minimum=list(self._flags_min.values()), 
            maximum=list(self._flags_max.values()), 
            name='observation'
        )

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
        while self._is_equal(self._state, self._target_location): 
            self._state = self._get_initial_state()
        
        self._best_state = self._state

        # self.ax.clear()

        # if self.render_mode == "human":
        #     self._render()

        if self._verbose: print(f"[RESET] state: {self._state}, target: {self._target_location}")

        return ts.restart(np.array(self._state[0], dtype=np.int32))

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()
        
        # print(type(action))
        # Apply an action: decrease/increase <var>
        self._action_mapping.get(int(action))()

        # We use `np.clip` to make sure we don't leave the boundaries
        for i in range(len(self._state[0])):
            # Iterate through each variable in `self._state`
            # and use `np.clip` to make sure we don't leave
            # the boundaries
            min_value_i = list(self._flags_min.values())[i]
            max_value_i = list(self._flags_max.values())[i]
            self._state[0][i] = np.clip(self._state[0][i], min_value_i, max_value_i)

        self._state[1] = self._y(self._state[0])

        # An episode is done if the agent has reached the target.
        self._episode_ended = self._is_equal(self._state, self._target_location)

        # A metric (y, throughput, latency, etc) is a reward itself
        # reward = -self._state[1]

        reward = 1 if self._episode_ended else 0
        
        # self.ax.clear()

        # if self.render_mode == "human":
        #     self._render_frame()

        if self._verbose: print(f"[STEP] state: {self._state}, target: {self._target_location}, reward: {reward}")

        if self._episode_ended:
            return ts.termination(
                np.array(self._state[0], dtype=np.int32), reward)
        else:
            return ts.transition(
                np.array(self._state[0], dtype=np.int32), reward=0.0, discount=1.0)

    def _y(self, params):
        x, y = params
        # z_min = z(0; 1) = -2
        return 3*x**2 + x*y + 2*y**2 - x - 4*y
    
    def _transform_action(self, action):
        # Convert action from [0, 1] to [-1, 1]
        return 2 * action - 1
    
    def _get_target(self):
        initial_guess = [1, 1]
        result = optimize.minimize(self._y, initial_guess)
        if result.success:
            target_arg = np.round(result.x, 5)
            target_f = np.round(result.fun, 5)
            # returns np.array([[x, y], z], dtype=np.int32)
            return np.array([target_arg, target_f], dtype=object)
        else:
            raise ValueError(result.message)
    
    def _get_initial_state(self):
        state_args = np.random.randint(
            list(self._flags_min.values()), 
            list(self._flags_max.values()),
            size=(self._num_variables,), 
            dtype=np.int32
        )
        state_f = self._y(state_args)
        return np.array([state_args, state_f], dtype=object)
    
    def _get_reward(self, next_state, current_state):
        """
        The reward is the relative difference between
        a current agent value and the next one. The 
        normalization puts a large measurement range on the
        same scale.
        
        Han, Xue & Yu, Tingting. (2020). Automated Performance Tuning for Highly-Configurable Software Systems. 
        """
        return (next_state - current_state) / current_state

    def _decrease_MaxHeapSize(self):
        self._state[0][0] -= 1
    
    def _increase_MaxHeapSize(self):
        self._state[0][0] += 1
    
    def _decrease_InitialHeapSize(self):
        self._state[0][1] -= 1
    
    def _increase_InitialHeapSize(self):
        self._state[0][1] += 1
    
    def _is_equal(self, state, target):
        return np.allclose(state[0], target[0])

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

class JVMEnv(py_environment.PyEnvironment):
    
    def __init__(
            self,
            jdk: str,
            bm_path: str,
            bm: str = "cassandra",
            n: int = 5,
            goal: str = "throughput",
            verbose: bool=False
        ):
        
        self._jdk = os.path.join(jdk, "bin")
        # self._state = 0
        self._episode_ended = False
        self._verbose = verbose
        self._bm = bm
        self._bm_path = bm_path
        self._gc_log_file = f"gc-{self._bm}.txt"
        self._n = n
        self._goal = goal

        self._env = os.environ.copy()
        self._env["PATH"] = f"{self._jdk}:{self._env['PATH']}"
        self._gc_viewer_jar = "gcviewer-1.36.jar"

        # TODO: Add more flags
        self._num_variables = 1
        # self._num_variables = 2
        self._flags = {
            "MaxHeapSize": {"min": 6.4e+7, "max": 4.29e+9},
            # "MaxHeapSize": {"min": 6.4e+7, "max": 1e+9}, # 64m to 1G
            # "InitialHeapSize": {"min": 1.25e+8, "max": 2.5e+8},
        }
        self._action_mapping = {
            0: self._decrease_MaxHeapSize,
            1: self._increase_MaxHeapSize,
            # 2: self._decrease_InitialHeapSize,
            # 3: self._increase_InitialHeapSize,
        }

        assert len(list(self._action_mapping.keys())) == 2*self._num_variables
        assert os.path.exists(self._gc_viewer_jar), f"{self._gc_viewer_jar} does not exist"
        assert os.path.exists(self._bm_path), f"{self._bm_path} does not exist"
        assert os.path.exists(self._jdk), f"{self._jdk} does not exist"
        
        self._flags_min_values = [self._flags[i]["min"] for i in self._flags.keys()]
        self._flags_max_values = [self._flags[i]["max"] for i in self._flags.keys()]

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), 
            dtype=np.int32, 
            minimum=0, 
            maximum=2*self._num_variables-1,  # {0, 1, 2, 3}
            name='action'
        )
        
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._num_variables,), 
            dtype=np.int64, 
            minimum=self._flags_min_values, 
            maximum=self._flags_max_values, 
            name='observation'
        )

        # It will take a while
        self._default_goal_value = self._get_initial_goal_value()
        self._default_state = self._get_initial_state()
        self._current_goal_value = self._default_goal_value

        """
        A cache to store performance
        measurements for states.
        Han, Xue & Yu, Tingting. (2020). 
        Automated Performance Tuning for Highly-Configurable Software Systems. 

        0: {"args": [10, 10], "goal": 234},
        1: {"args": [10, 20], "goal": 222},
        """
        self._perf_states = {
            0: {"args": self._default_state[0], "goal": self._default_goal_value}
        }

        
    def action_spec(self):
        return self._action_spec
    
    def observation_spec(self):
        return self._observation_spec
    
    def _reset(self):
        self._episode_ended = False
        self._current_goal_value = self._default_goal_value
        self._state = self._default_state

        # self.ax.clear()
        # if self.render_mode == "human":
        #     self._render()

        logging.debug(f"[RESET] state: {self._state}, target: {self._current_goal_value}")
        return ts.restart(np.array(self._state[0], dtype=np.int64))

    def _step(self, action):
        if self._episode_ended:
            # The last action ended the episode. Ignore the current action and start
            # a new episode.
            return self.reset()
        # Apply an action based on the mapping: decrease/increase <flag_value>
        new_state = self._action_mapping.get(int(action))(self._state)
        # Make sure we don't leave the boundaries
        self._state = self._clip_state(new_state)
        
        # Get new JVM options values after setting `self._state`
        jvm_opts = self._update_jvm_opts()
        # Launch a benchmark with a new JVM configuration
        self._run(jvm_opts, self._gc_log_file, self._bm, self._bm_path, self._n)
        # Get the goal value
        goal = self._get_goal_value()
        
        previous_goal_value = self._current_goal_value
        
        self._state[1] = goal
        self._current_goal_value = goal

        # self._episode_ended = self._is_equal(self._state, self._target_location)
        if self._current_goal_value <= self._default_goal_value // 2:
            self._episode_ended = True

        # ! Multiply by (-1) if lower is better
        # reward = -1 * self._current_goal_value 
        # reward = self._get_reward(self._current_goal_value, previous_goal_value)
        reward = self._get_reward(self._current_goal_value, self._default_goal_value)
        
        # self.ax.clear()

        # if self.render_mode == "human":
        #     self._render_frame()

        logging.debug(f"[STEP] state: {self._state}, current_goal_value: {self._current_goal_value}, reward: {reward}")

        if self._episode_ended:
            return ts.termination(
                np.array(self._state[0], dtype=np.int64), reward)
        else:
            return ts.transition(
                np.array(self._state[0], dtype=np.int64), reward=0.0, discount=1.0)
    
    def _state_merging(self, flags):
        """
        Store states' JVM configurations and performance measurements
        in a cache. The cache stores the JVM flags and the metric goal.
        In each `self._step` iteration, the performance of the same
        state is queried and retrieved directly from the cache instead
        of re-running the benchmark utility.
        """
        logging.debug("[_STATE_MERGING]", self._perf_states)
        saved_states = [self._perf_states[i]["args"] for i in self._perf_states.keys()]
        logging.debug("[_STATE_MERGING]", self._perf_states)
        if flags in saved_states:
            for i in self._perf_states.keys():
                """ 
                If current state is stored in a cache,
                update the state goal value.
                """
                if flags == self._perf_states[i]["args"]:
                    goal = self._perf_states[i]["goal"]
        elif flags not in saved_states:
            """ 
            If current state is not stored in the cache,
            measure the performance metric, and save it
            in the cache.
            """
            try:
                last_index = list(self._perf_states.keys())[-1]
            except IndexError:
                # If `self._perf_states` is empty
                last_index = -1

            # Get new JVM options values after setting `self._state`
            jvm_opts = self._update_jvm_opts()
            # Launch a benchmark with a new JVM configuration
            self._run(jvm_opts, self._gc_log_file, self._bm, self._bm_path, self._n)
            # Get the goal value
            goal = self._get_goal_value()
            # Store a new state in the cache
            self._perf_states[last_index + 1] = {"args": flags, "goal": goal}
        return flags, goal

    def _get_JVM_opt_value(self, opt: str):
        """
        Get the defaul JVM option value from environment
        by parsing java PrintFlagsFinal output.

        Parameters:
        opt (str): JVM option name

        Returns:
        (int) Default JVM option value
        """
        flags_process = subprocess.Popen(
            ["java", "-XX:+PrintFlagsFinal", "-version"],
            stdout=subprocess.PIPE,
            text=True,
            env=self._env)

        grep_process = subprocess.Popen(
            ["grep", opt, "-m", "1"],
            stdin=flags_process.stdout,
            stdout=subprocess.PIPE,
            text=True)

        output, error = grep_process.communicate()
        result = re.findall("\d+", output)[0]

        return int(result)
    
    def _get_initial_goal_value(self):
        """
        Run the benchmark with default values
        and get a start point of the goal.
        """
        # Run benchmark with default values
        self._run([], self._gc_log_file, self._bm, self._bm_path, self._n)

        if os.path.exists(self._gc_log_file):
            # Get goal value from first-time generated GC log
            result = self._get_goal_value()
            # Clean up
            os.remove(self._gc_log_file)
            return result
        else:
            raise Exception(
                "GC log file was not generated: please, ensure that benchmark runs correctly!")
    
    def _get_initial_state(self):
        flags = []
        for flag_name in self._flags.keys():
            flags.append(self._get_JVM_opt_value(flag_name))
        # goal = self._get_initial_goal_value()
        goal = self._default_goal_value
        initial_state = np.array([flags, goal], dtype=object)
        return initial_state
    
    def _clip_state(self, state):
        for i in range(len(state[0])):
            """ 
            Iterate through each flag value in `state`
            and use `np.clip` to make sure we don't leave
            the boundaries
            """
            min_value_i = self._flags_min_values[i]
            max_value_i = self._flags_max_values[i]
            state[0][i] = np.clip(state[0][i], min_value_i, max_value_i)
        return state
    
    def _get_goal_value(self):
        
        sep = ';'
        summary = "summary.csv"
        goal_value = None
        subprocess.call(
            ["java",
            "-cp", self._gc_viewer_jar,
            "com.tagtraum.perf.gcviewer.GCViewer",
            self._gc_log_file,
            summary, 
            "-t", "SUMMARY"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text = True,
            env = self._env)
        
        assert os.path.exists(summary), f"File {summary} does not exist"

        with open(summary, "+r") as summary_file:
            for line in summary_file.readlines():
                if self._goal + sep in line:
                    goal_value = float(line.split(sep)[1])
            
        if goal_value == None:
            raise Exception(f"Goal '{self._goal}' was not found in '{summary}'!")
                    
        return goal_value
    
    def _get_reward(self, next_state, current_state):
        """
        The reward is the relative difference between
        a current agent value and the next one. The 
        normalization puts a large measurement range on the
        same scale.
        
        Han, Xue & Yu, Tingting. (2020). Automated Performance Tuning for Highly-Configurable Software Systems. 
        """
        return (next_state - current_state) / current_state

    def _decrease_MaxHeapSize(self, state):
        coef = 100 * (2**10) * (2**10) # 100 Mb
        state[0][0] -= 1 * coef
        return state
    
    def _increase_MaxHeapSize(self, state):
        coef = 100 * (2**10) * (2**10) # 100 Mb
        state[0][0] += 1 * coef
        return state
    
    def _decrease_InitialHeapSize(self, state):
        coef = 100 * (2**10) * (2**10) # 100 Mb
        state[0][1] -= 1 * coef
        return state
    
    def _increase_InitialHeapSize(self, state):
        coef = 100 * (2**10) * (2**10) # 100 Mb
        state[0][1] += 1 * coef
        return state
    
    def _is_equal(self, state, target):
        return np.allclose(state[0], target[0])
    
    def _update_jvm_opts(self):
        jvm_opts = []
        for flag_name in self._flags.keys():
            i = list(self._flags.keys()).index(flag_name)
            # `self._state[0]` stores an array of current
            # flag values [<MaxHeapSize>, <InitialHeapSize>]
            flag_value = int(self._state[0][i])
            # TODO: Add a condition for flags with boolean values
            jvm_opts.append(f"-XX:{flag_name}={flag_value}")
        return jvm_opts

    def _run(
            self, 
            jvm_opts: List[str], 
            gc_log_file: str, 
            bm: str, 
            bm_path: str, 
            n: int=5, 
            verbose: bool=False):
        """
        Run a benchmark with the specified JVM options
        such as MaxHeapSize.

        Parameters:
        jvm_opts (array of str):  JVM options (e.g., MaxHeapSize, GCTimeRation,
                                ParallelGCThreads, etc).
        gc_log_file (str):        Path to the garbage collector log file (txt).
        bm (str):                 DaCapo benchmark name.
        bm_path (str):            DaCapo benchmark path.
        n (int):                  Total number of benchmark's iterations.
        verbose (bool):           Print debug messages.
        """

        # Clean up before running the benchmark
        if os.path.exists(gc_log_file): os.remove(gc_log_file)

        jvm_opts.append("-XX:+UseParallelGC")

        logging.debug("Running benchmark...")

        # Run the benchmark (hide output)
        try:
            logging.debug(f"Running {bm} with JVM_OPTS={jvm_opts}")
            subprocess.check_output(
                ["java",
                "-cp", bm_path,
                f"-Xlog:gc*=trace:file={gc_log_file}:tags,time,uptime,level",
                *jvm_opts,
                "-Dvmstat.enable_jfr=yes",
                "-Dvmstat.csv=yes", "Harness", "-v", "-n", f"{n}", bm],
                stderr=subprocess.STDOUT,
                text = True,
                env = self._env)
        except subprocess.CalledProcessError as e:
            raise Exception("Command failed with return code", e.returncode, e.output)

        return
    
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