import copy
import logging
import os
import random
import re
import subprocess
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types


class JVMEnv(py_environment.PyEnvironment):
    def __init__(
        self,
        jdk_path: str,
        bm_path: str,
        gc_viewer_jar: str,
        callback_path: str,
        bm_name: str = "avrora",
        n: int = 5,
        goal: str = "avgGCPause",
        verbose: bool = False,
    ):
        """A Python environment class for Reinforcement Learning
        algorithm purposes. It represents a utility class that can be
        used for a Java Virtual Machine (JVM) configuration tuning.

        Args:
            jdk_path (str): A path to JDK directory.
            bm_path (str): A path to DaCapo benchmark jar file to run.
            gc_viewer_jar (str): A path to GCViewer jar file.
            callback_path (str): A path to DaCapo benchmark's callback file.
            bm_name (str, optional): DaCapo benchmark name. Defaults to "avrora".
            n (int, optional): A number of iterations to run the benchmark. Defaults to 5.
            goal (str, optional): A user's objective name they want to tune.
                The objective's name is extracted from a summary file (GCViewer
                results). Defaults to "avgGCPause".
            verbose (bool, optional): Print debug messages. Defaults to False.
        """

        self.jdk = jdk_path
        self.bm_path = bm_path
        self.gc_viewer_jar = gc_viewer_jar
        self._callback_path = callback_path
        self._bm = bm_name
        self._gc_log_file = f"gc-{self._bm}.txt"
        self._n = n
        self._goal = goal
        self._verbose = verbose

        self._env = os.environ.copy()
        self._env["PATH"] = f"{self.jdk}:{self._env['PATH']}"
        self._episode_ended = False

        # ============= F L A G S =============
        # TODO: Add more flags
        # TODO: Find min and max values for flags
        self._num_variables = 2
        self._goal_idx = self._num_variables
        self._flags = {
            "MaxTenuringThreshold": {"min": 1, "max": 16, "step": 3},
            "ParallelGCThreads": {"min": 4, "max": 24, "step": 4},
        }

        self._action_mapping = {
            0: self._decrease_MaxTenuringThreshold,
            1: self._increase_MaxTenuringThreshold,
            2: self._decrease_ParallelGCThreads,
            3: self._increase_ParallelGCThreads,
        }
        # =====================================

        assert (
            len(self._action_mapping) == 2 * self._num_variables
        ), "Each flag should have 2 actions!"

        self._flags_min_values = [self._flags[i]["min"] for i in self._flags.keys()]
        self._flags_max_values = [self._flags[i]["max"] for i in self._flags.keys()]
        self._flags_step_values = [self._flags[i]["step"] for i in self._flags.keys()]

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(),
            dtype=np.int32,
            minimum=0,
            maximum=2 * self._num_variables - 1,  # {0, 1, 2, 3}
            name="action",
        )

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._num_variables + 1 + 7,),  # 1 goal, {X} external vars
            dtype=np.float32,
            name="observation",
        )

        # For offline RL: if you already have a dataset file with trajectories.
        self._new_df = pd.read_csv(f"datasets/{self._bm}_real_saved_states.csv")

        # self._default_state = self._get_default_state(mode="default")
        self._default_state = self._get_default_state(mode="random")

        self._perf_states = {}
        self._perf_states[0] = {
            "args": self._default_state[: self._num_variables],
            "goal": self._default_state[self._goal_idx],
            "extra": self._default_state[self._goal_idx + 1 :],
            "count": 1,
        }

        self._print_welcome_msg()

    @property
    def jdk(self):
        return self._jdk

    @jdk.setter
    def jdk(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self._jdk = os.path.join(path, "bin")

    @property
    def bm_path(self):
        return self._bm_path

    @bm_path.setter
    def bm_path(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self._bm_path = os.path.join(path)

    @property
    def gc_viewer_jar(self):
        return self._gc_viewer_jar

    @gc_viewer_jar.setter
    def gc_viewer_jar(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self._gc_viewer_jar = os.path.join(path)

    def action_spec(self):
        """Get the actions that should be provided to `step()`."""
        return self._action_spec

    def observation_spec(self):
        """Get the the observations provided by the environment."""
        return self._observation_spec

    @property
    def performance_states(self) -> Dict[int, Any]:
        return self._perf_states

    def _reset(self):
        """
        Resets the environment state.

        This method must be called before :func:`step()`.

        Returns:
            A `TimeStep` namedtuple containing:
                step_type: A `StepType` of `FIRST`.
                reward: 0.0, indicating the reward.
                discount: 1.0, indicating the discount.
                observation: A NumPy array, or a nested dict,
                list or tuple of arrays
                corresponding to `observation_spec()`.
        """
        self._episode_ended = False

        # To ensure all elements within an object array are copied, use `copy.deepcopy`
        self._default_state = self._get_default_state(mode="random")
        # self._default_state = self._get_default_state(mode="default")
        self._state = copy.deepcopy(self._default_state)

        logging.debug(
            f"[RESET] {self._get_info()}, target: {self._state[self._goal_idx]}"
        )
        return ts.restart(np.array(self._state, dtype=np.float32))

    def _step(self, action: types.NestedArray):
        """Updates the environment according to action.

        Parameters:
            action: A NumPy array, or a nested dict, list or tuple of arrays
                corresponding to `action_spec()`.
        Returns:
            A `TimeStep` namedtuple containing:
                step_type: A `StepType` of `FIRST`.
                reward: 0.0, indicating the reward.
                discount: 1.0, indicating the discount.
                observation: A NumPy array, or a nested dict, list or tuple of arrays
                corresponding to `observation_spec()`.
        """
        if self._episode_ended:
            # The last action ended the episode.
            # Ignore the current action and start a new episode.
            logging.debug(
                f"[EPISODE ENDED] {self._get_info()}, target: {self._state[self._goal_idx]}"
            )
            return self.reset()
        # Apply an action based on the mapping: decrease/increase <flag.value>.
        self._action_mapping.get(int(action))()

        # Make sure we don't leave the boundaries.
        self._state = self._clip_state(self._state)

        # Check if the current JVM configuration is cached.
        # Add `state` to cache if new.
        self._state = copy.deepcopy(
            self._state_merging(self._state[: self._num_variables])
        )

        # Termination criteria
        if self._state[self._goal_idx] <= self._default_state[self._goal_idx] * 0.04:
            self._episode_ended = True

        self._reward = self._get_reward(
            current_state=self._state,
            previous_state=self._default_state,
            lower_is_better=True,
            beta=0.0,
        )  # ! No intrinsic reward

        logging.debug(
            f"[STEP] {self._get_info()}, target: {self._state[self._goal_idx]}"
        )

        if self._episode_ended:
            return ts.termination(np.array(self._state, dtype=np.float32), reward=2.0)
        else:
            return ts.transition(
                np.array(self._state, dtype=np.float32),
                reward=self._reward,
                discount=0.5,
            )

    def _state_merging(self, flags):
        """
        Store states' JVM configurations and performance measurements
        in a cache. The cache stores the JVM flags and the metric goal.
        In each `self._step` iteration, the performance of the same
        state is queried and retrieved directly from the cache instead
        of re-running the benchmark utility.
        Han, Xue & Yu, Tingting. (2020).
        Automated Performance Tuning for Highly-Configurable Software Systems.

        0: {"args": [10, 10], "goal": 234, "extra": [150, 0.1, 30, 19], "count": 1},
        1: {"args": [10, 20], "goal": 222, "extra": [140, 0.2, 29, 18], "count": 4},
        ...
        """
        saved_states = [self._perf_states[i]["args"] for i in self._perf_states.keys()]
        state = []
        if flags == self._default_state[: self._num_variables]:
            state = self._default_state
        elif flags in saved_states:
            """
            If current state is stored in a cache,
            update the state goal value.
            """
            idx = saved_states.index(flags)
            self._perf_states[idx]["count"] += 1
            goal = self._perf_states[idx]["goal"]
            extra = self._perf_states[idx]["extra"]
            state = [*flags, goal, *extra]
        else:
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
            # Launch a benchmark with a new JVM configuration
            state = self._synthetic_run(flags)
            # Store a new state in the cache, count = 1.
            self._perf_states[last_index + 1] = {
                "args": list(flags),
                "goal": state[self._goal_idx],
                "extra": state[self._goal_idx + 1 :],
                "count": 1,
            }
        assert len(state) != 0
        return state

    def _synthetic_run(self, flag_values: List[int]):
        flag_names = list(self._new_df.columns[: self._num_variables])
        param_names = list(self._new_df.columns[self._num_variables :])
        # Select columns where flags equal to flag_values
        param_values = (
            self._new_df.set_index(flag_names)
            .loc[tuple(flag_values), param_names]
            .values
        )
        state = [*flag_values, *param_values]
        return state

    def _get_JVM_opt_value(self, opt: str):
        """
        Get the defaul JVM option value from environment
        by parsing java PrintFlagsFinal output.

        Parameters:
        opt (str): JVM option name

        Returns:
        (int) Default JVM option value
        """
        opt_value = None

        flags = subprocess.check_output(
            ["java", "-XX:+PrintFlagsFinal", "-version"], text=True, env=self._env
        )

        assert re.search(opt, flags), f"Option {opt} was not found in JVM flags"

        for line in flags.split("\n"):
            if re.search(opt, line):
                opt_value = re.findall("\d+", line)[0]
                # TODO: Handle boolean values
                return int(opt_value)

    def _get_info(self):
        info = {}
        flags = list(self._flags.keys())
        for flag in flags:
            flag_idx = flags.index(flag)
            flag_value = self._state[flag_idx]
            info[flag] = flag_value
        return info

    def _get_default_state(self, mode: str = "default"):
        """
        Get default values for each JVM option stored in `self._flags`
        from `java -XX:+PrintFlagsFinal -version` command output.
        Also runs a benchmark with default JVM_OPTS and get the initial
        goal value.

        Parameters:
        mode (str): The mode of generating JVM options (default/minimum/random)
                    where
                    - "default" sets JVM_OPTS default JDK options,
                    - "minimum" sets JVM_OPTS minimum possible values,
                    - "random" sets JVM_OPTS random values within options
                    range.

        Returns:
        (np.array) The initial state of the Agent which stores the default
        JVM_OPTS and its performance measurement.
        """
        assert mode in ["default", "random"], f"Unknown mode: {mode}"

        if mode == "default":
            state = self._synthetic_run([7, 12])
        if mode == "random":
            rand_flags = []
            for i in range(self._num_variables):
                min_val = self._flags_min_values[i]
                max_val = self._flags_max_values[i]
                step_val = self._flags_step_values[i]
                rand_flags.append(
                    random.randrange(
                        start=min_val,
                        stop=max_val + step_val,  # must be included
                        step=step_val,
                    )
                )
            state = self._synthetic_run(rand_flags)
        return state

    def _get_goal_value(self, jvm_opts: List[str] = []):
        """
        Run the benchmark with default values and
        get a start point of the goal.

        Parameters:
        jvm_opts (list): An array of JVM options.

        Returns:
        (int) A performance measurement (target metric
        or goal value).
        """
        # Run benchmark with default values
        self._run(
            jvm_opts,
            self._gc_log_file,
            self._bm,
            self.bm_path,
            self._callback_path,
            self._n,
        )

        if os.path.exists(self._gc_log_file):
            # Get goal value from first-time generated GC log
            result = self._get_goal_from_file()
            # Clean up
            os.remove(self._gc_log_file)
            return result
        else:
            raise Exception(
                "GC log file was not generated: please, ensure that benchmark runs correctly!"
            )

    def _clip_state(self, state):
        for i in range(self._num_variables):
            """
            Iterate through each flag value in `state`
            and use `np.clip` to make sure we don't leave
            the boundaries
            """
            min_value_i = self._flags_min_values[i]
            max_value_i = self._flags_max_values[i]
            state[i] = np.clip(state[i], min_value_i, max_value_i)
        return state

    def _get_goal_from_file(self):
        sep = ";"
        summary = "summary.csv"
        goal_value = None
        subprocess.call(
            [
                "java",
                "-cp",
                self.gc_viewer_jar,
                "com.tagtraum.perf.gcviewer.GCViewer",
                self._gc_log_file,
                summary,
                "-t",
                "SUMMARY",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text=True,
            env=self._env,
        )

        assert os.path.exists(summary), f"File {summary} does not exist"

        with open(summary, "+r") as summary_file:
            for line in summary_file.readlines():
                if self._goal + sep in line:
                    goal_value = float(line.split(sep)[1])

        if goal_value == None:
            raise Exception(f"Goal '{self._goal}' was not found in '{summary}'!")

        if os.path.exists(summary):
            os.remove(summary)

        return goal_value

    def _get_reward(
        self,
        current_state: np.array,
        previous_state: np.array,
        lower_is_better: bool = False,
        beta: float = 1.0,
    ):
        """
        Get the environment reward. The reward is composed of two terms:
            `reward = reward_ex + beta * reward_in`,
        where `beta` is a hyperparameter adjusting the balance between
        exploitation and exploration.
        - reward_ex is an extrinsic reward from the environment at time `t`.
        - reward_in is an intrinsic exploration bonus.
        Parameters:
            current_state (np.array):   Current state containing JVM flags
                                        and goal value.
            previous_state (np.array):  Previous state containing JVM flags
                                        and goal value.
            lower_is_better (bool):     Whether to consider lower goal values
                                        better than larger ones. So that reward
                                        is positive.
            beta (float):               Importance of instrinsic rewards ([0.0, 1.0]).
        Returns:
            reward (float):             An environment reward value at current
                                        time step.
        """
        coef = 1
        reward_in = 0  # Intrinsic reward.
        reward_ex = 0  # Extrinsic reward.

        # TODO: Check if beta is in range [0, 1]

        if lower_is_better:
            coef = -1

        current_goal = current_state[self._goal_idx]
        previous_goal = previous_state[self._goal_idx]

        if coef * current_goal <= previous_goal:
            for i in self._perf_states.keys():
                if self._perf_states[i]["args"] == current_state[: self._num_variables]:
                    # First, we add the state to cache with count=1,
                    # but we haven't run the benchmark with it yet,
                    # so it is fair to say that actually count is 0.
                    count = self._perf_states[i]["count"] - 1
                    reward_in = (count + 0.01) ** (-1 / 2)
            reward_ex = coef * (current_goal - previous_goal) / previous_goal
            reward = reward_ex + beta * reward_in
            reward = round(reward, 4)
        else:
            reward = -1

        return reward

    def _decrease_MaxTenuringThreshold(self):
        idx = 0
        coef = 3
        saved_values = [
            self._perf_states[i]["args"][idx] for i in self._perf_states.keys()
        ]
        self._state[idx] = min(
            saved_values, key=lambda x: abs(x - (self._state[idx] - coef))
        )
        self._state[idx] -= coef

    def _increase_MaxTenuringThreshold(self):
        idx = 0
        coef = 3
        saved_values = [
            self._perf_states[i]["args"][idx] for i in self._perf_states.keys()
        ]
        self._state[idx] = min(
            saved_values, key=lambda x: abs(x - (self._state[idx] + coef))
        )
        self._state[idx] += coef

    def _decrease_ParallelGCThreads(self):
        idx = 1
        coef = 4
        saved_values = [
            self._perf_states[i]["args"][idx] for i in self._perf_states.keys()
        ]
        self._state[idx] = min(
            saved_values, key=lambda x: abs(x - (self._state[idx] - coef))
        )
        self._state[idx] -= coef

    def _increase_ParallelGCThreads(self):
        idx = 1
        coef = 4
        saved_values = [
            self._perf_states[i]["args"][idx] for i in self._perf_states.keys()
        ]
        self._state[idx] = min(
            saved_values, key=lambda x: abs(x - (self._state[idx] + coef))
        )
        self._state[idx] += coef

    # def _is_equal(self, state, target):
    #     return np.allclose(state[0], target[0])

    def _get_jvm_opts(self, flags):
        jvm_opts = []
        for flag_name in self._flags.keys():
            i = list(self._flags.keys()).index(flag_name)
            # `state[:self._num_variables]` stores an array of current
            # flag values [<MaxHeapSize>, <InitialHeapSize>]
            flag_value = int(flags[i])
            # TODO: Add a condition for flags with boolean values
            jvm_opts.append(f"-XX:{flag_name}={flag_value}")
        return jvm_opts

    def _run(
        self,
        jvm_opts: List[str],
        gc_log_file: str,
        bm: str,
        bm_path: str,
        callback_path: str,
        n: int = 5,
        verbose: bool = False,
    ):
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

        if not os.path.exists(callback_path):
            raise FileNotFoundError(callback_path)

        # Clean up before running the benchmark
        if os.path.exists(gc_log_file):
            os.remove(gc_log_file)

        # Default flags
        jvm_opts.append("-XX:+UseParallelGC")
        jvm_opts.append("-Xmx16G")
        jvm_opts.append("-Xms16G")
        jvm_opts.append("-XX:SurvivorRatio=130")
        jvm_opts.append("-XX:TargetSurvivorRatio=66")

        # Run the benchmark (hide output)
        try:
            subprocess.check_output(
                [
                    "java",
                    "-cp",
                    f"{callback_path}:{bm_path}",
                    f"-Xlog:gc*=trace:file={gc_log_file}:tags,time,uptime,level",
                    *jvm_opts,
                    "-Dvmstat.enable_jfr=yes",
                    "-Dvmstat.csv=yes",
                    "Harness",
                    "-v",
                    "-n",
                    f"{n}",
                    bm,
                ],
                stderr=subprocess.STDOUT,
                text=True,
                env=self._env,
            )
        except subprocess.CalledProcessError as e:
            raise Exception("Command failed with return code", e.returncode, e.output)

        return

    def _render(self):
        raise NotImplementedError("This environment has not implemented `render().'")

    def _print_welcome_msg(self):
        print(
            "Successfully initialized a JVM Environment!\n",
            f"JDK: {self.jdk},\n",
            f"Benchmark: {self._bm} ({self.bm_path}),\n",
            f"Number of iterations: {self._n},\n",
            f"Goal: {self._goal},\n",
            f"Number of JVM options: {self._num_variables},\n",
            f"JVM options: {self._flags},\n",
            f"Env. default state: {self._default_state},\n",
            f"Env. default goal value: {self._default_state[self._goal_idx]},\n",
        )
