import copy
import logging
import os
import random
import re
import subprocess
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types


class JVMState:
    def __init__(self, flags=[], params=[], goals=[]):
        self._flags = flags
        self._params = params
        self._goals = goals

    def get_list(self):
        return np.array([*self._flags, *self._params, *self._goals])

    def get_flags(self):
        return self._flags

    def get_params(self):
        return self._params

    def get_goals(self):
        return self._goals

    def set_flags(self, flags):
        self._flags = flags

    def set_params(self, params):
        self._params = params

    def set_goals(self, goals):
        self._goals = goals


class JVMEnv(py_environment.PyEnvironment):
    """A PyEnvironment for Java Virtual Machine optimization."""

    def __init__(
        self,
        jdk_path: str,
        bm_path: str,
        gc_viewer_jar: str,
        callback_path: str,
        goals: List,
        flags: List,
        params: List,
        default_flags: List,
        bm_name: str = "avrora",
        n: int = 5,
        verbose: bool = False,
        beta: float = 0.0,
        run_offline: bool = True,
    ):
        """Create a JVM Python Envrionment by passing an OpenJDK,
        DaCapo benchmark executive jar, DaCapo benchmark name,
        GCViewer jar to parse Garbage Collector logging files,
        and other options to run a benchmark, measure the performance,
        and the JVM flags to configure the next run.

        Args:
            jdk_path (str): An OpenJDK directory.
            bm_path (str): A path to DaCapo benchmark jar.
            gc_viewer_jar (str): A path to GCViewer jar.
            callback_path (str): A path to DaCapo callback file.
            goals (List): A list of user performance metrics in such format:
            ```
            [
                {
                    "name": "footprint",
                    "minimize": true
                }
            ]
            ```
            flags (List): A list of JVM flags to consider when tuning:
            ```
            [
                {
                    "name": "MaxTenuringThreshold",
                    "min": 1,
                    "max": 16,
                    "step": 3
                },
                {
                    "name": "ParallelGCThreads",
                    "min": 4,
                    "max": 24,
                    "step": 4,
                    "unit": ""
                },
                {
                    "name": "MaxHeapSize",
                    "min": 2,
                    "max": 24,
                    "step": 4,
                    "unit": "G"
                }
            ],
            ```
            params (List): A list of external parameters to consider when tuning.
                These are extracted from GCViewer summary file.
            ```
            [
                "fullGCPause",
                "avgPause",
                "fullGcPauseCount",
                "footprint",
                "gcPerformance",
                "totalTenuredUsedMax",
                "avgPromotion",
                "fullGCPerformance"
            ]
            ```
            default_flags (List): JVM flags that will be used in every run of a benchmark.
            ```
            ["-XX:+UseParallelGC"]
            ```
            bm_name (str, optional): DaCapo benchmark name. Defaults to "avrora".
            n (int, optional): Number of iterations to run a benchmark. Defaults to 5.
            verbose (bool, optional): Print debug messages. Defaults to False.
            beta (float, optional): Importance of intrinsic rewards ([0.0, 1.0]). Defaults to 0.0.
        """
        self.jdk = jdk_path
        self.bm_path = bm_path
        self.gc_viewer_jar = gc_viewer_jar
        self.callback_path = callback_path
        self.bm_name = bm_name
        self._gc_log_file = f"gc-{self.bm_name}.txt"
        self._n = n
        self._goals = goals
        self._flags = flags
        self._params = params
        self._default_flags = default_flags
        self._verbose = verbose
        self._beta = beta
        self._run_offline = run_offline

        self._env = os.environ.copy()
        self._env["PATH"] = f"{self.jdk}:{self._env['PATH']}"

        self._episode_ended = False

        # ============= F L A G S =============
        self._num_flags = len(self._flags)
        self._num_params = len(self._params)
        self._num_goals = len(self._goals)

        # TODO: #9 Add support of boolean JVM flags
        self._flags_min_values = [each["min"] for each in self._flags]
        self._flags_max_values = [each["max"] for each in self._flags]
        self._flags_step_values = [each["step"] for each in self._flags]
        self._goals_optimization_target = [each["minimize"] for each in self._goals]

        self._flags_names = [each["name"] for each in self._flags]
        self._goals_names = [each["name"] for each in self._goals]
        self._params_names = self._params

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(self._num_flags,),
            dtype=np.int32,
            minimum=np.array(self._flags_min_values),
            maximum=np.array(self._flags_max_values),
            name="action",
        )

        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(self._num_flags + self._num_goals + self._num_params,),
            dtype=np.float32,
            name="observation",
        )

        self._default_state = self._get_default_state(mode="default")
        # self._default_state = self._get_default_state(mode="random")

        if run_offline:
            # For offline RL: if you already have a dataset file with saved states.
            self._cache_filename = (
                f"datasets/large_{self.bm_name}_real_saved_states.csv"
            )
            self._cached_states_df = pd.read_csv(self._cache_filename)
            self._perf_states = self._load_states_from_cache()
        else:
            self._perf_states = {}
            self._perf_states[0] = {
                "flags": self._default_state.get_flags(),
                "goals": self._default_state.get_goals(),
                "params": self._default_state.get_params(),
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
    def bm_name(self):
        return self._bm_name

    @bm_name.setter
    def bm_name(self, name: str):
        if name in ["avrora", "kafka", "h2"]:
            self._bm_name = name
        else:
            raise Exception(f"No such DaCapo benchmark: {name}")

    @property
    def gc_viewer_jar(self):
        return self._gc_viewer_jar

    @gc_viewer_jar.setter
    def gc_viewer_jar(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self._gc_viewer_jar = os.path.join(path)

    @property
    def callback_path(self):
        return self._callback_path

    @callback_path.setter
    def callback_path(self, path: str):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        self._callback_path = os.path.join(path)

    def action_spec(self):
        """Get the actions that should be provided to `step()`."""
        return self._action_spec

    def observation_spec(self):
        """Get the the observations provided by the environment."""
        return self._observation_spec

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

        self._state = copy.deepcopy(self._default_state)

        self._print_info("reset")

        return ts.restart(np.array(self._state.get_list(), dtype=np.float32))

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
            self._print_info("episode ended")
            return self.reset()

        self._state.set_flags(action.action)

        # Check if the current JVM configuration is cached.
        # If new, run benchmark, measure performance,
        # and add this state to cache.
        self._state = self._state_merging(self._state.get_flags())

        self._reward = self._get_reward(
            current_state=self._state,
            previous_state=self._default_state,
            lower_is_better=self._goals_optimization_target,
            beta=self._beta,
        )

        self._print_info("step")

        return ts.transition(
            np.array(self._state.get_list(), dtype=np.float32),
            reward=self._reward,
            discount=0.5,
        )

    def save(self, filename: str):
        """Save performance states to CSV file.

        Args:
            filename (str): Destination file path.
        """
        if len(self._perf_states):
            columns = [*self._flags_names, *self._params_names, *self._goals_names]
            states = np.array(
                [
                    [*state["flags"], *state["params"], *state["goals"]]
                    for state in self._perf_states.values()
                ]
            )
            logging.info("Saving environment states to file %s".format(filename))
            cache_df = pd.DataFrame(states, columns=columns)
            # Append to file if exists.
            use_header = not os.path.exists(filename)
            cache_df.to_csv(filename, index=False, mode="a", header=use_header)
        else:
            warnings.warn("Cache is empty. Nothing to backup...")

    def _state_merging(self, flags):
        """
        Store states' JVM configurations and performance measurements
        in a cache. The cache stores the JVM flags and the metric goal.
        In each `self._step` iteration, the performance of the same
        state is queried and retrieved directly from the cache instead
        of re-running the benchmark utility.
        Han, Xue & Yu, Tingting. (2020).
        Automated Performance Tuning for Highly-Configurable Software Systems.

        0: {"flags": [10, 10], "goal": [234], "params": [150, 0.1, 30, 19], "count": 1},
        1: {"flags": [10, 20], "goal": [222], "params": [140, 0.2, 29, 18], "count": 4},
        ...
        """
        saved_states = [self._perf_states[i]["flags"] for i in self._perf_states.keys()]
        state = JVMState()
        if np.array_equal(flags, self._default_state.get_flags()):
            state = copy.deepcopy(self._default_state)
        elif np.any(np.all(flags == saved_states, axis=1)):
            """
            If current state is stored in a cache,
            update the state goal value.
            """
            # idx = saved_states.index(flags)
            idx = np.where((saved_states == flags).all(axis=1))[0][0]
            self._perf_states[idx]["count"] += 1

            goal = self._perf_states[idx]["goals"]
            params = self._perf_states[idx]["params"]

            state = JVMState(flags, params, goal)
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
            # state = self._synthetic_run(flags)
            state = self._run(flags)

            # Store a new state in the cache, count = 1.
            self._perf_states[last_index + 1] = {
                "flags": state.get_flags(),
                "goals": state.get_goals(),
                "params": state.get_params(),
                "count": 1,
            }

        assert len(state.get_list()) != 0, "State is empty!"

        return state

    def _synthetic_run(self, flags_values: List[int]):
        try:
            # Select columns where flags equal to flag_values
            params_values = (
                self._cached_states_df.set_index(self._flags_names)
                .loc[tuple(flags_values), self._params_names]
                .values
            )

            goals_values = (
                self._cached_states_df.set_index(self._flags_names)
                .loc[tuple(flags_values), self._goals_names]
                .values
            )
        except KeyError:
            warnings.warn("No configuration found in cache. Generating random numbers")
            params_values = np.random.random((self._num_params,))
            goals_values = np.random.random((self._num_goals,))

        state = JVMState(
            flags=flags_values,
            params=params_values,
            goals=goals_values,
        )
        return state

    def _get_default_jvm_option_value(self, opt: str):
        """
        Get the default JVM option value from environment
        by parsing java PrintFlagsFinal output.
        Equals to bash command:
        java -XX:+PrintFlagsFinal -version|grep "$opt"|

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
                # TODO: #9 Add support of boolean JVM flags
                return int(opt_value)

    def _load_states_from_cache(self) -> Dict:
        """Create a dictionary of performance states,
        load saved variables from a cache dataframe.

        Returns:
            Dict: Loaded from cache dataframe performance states.
        """
        flags = self._cached_states_df[self._flags_names].values
        params = self._cached_states_df[self._params_names].values
        goals = self._cached_states_df[self._goals_names].values

        try:
            perf_states = {}
            num_states = self._cached_states_df.shape[0]
            for i in range(num_states):
                perf_states[i] = {
                    "flags": flags[i],
                    "params": params[i],
                    "goals": goals[i],
                    "count": 1,
                }
        except IndexError:
            logging.error(
                "Could not load perf states from a cache pd.DataFrame! \
                    Perhaps there is no states in the dataframe."
            )
        return perf_states

    def _print_info(self, stage: str):
        info = {}
        for i, flag_info in enumerate(self._flags):
            flag_name = flag_info["name"]
            flag_value = self._state.get_flags()[i]
            info[flag_name] = flag_value
        logging.debug("[{stage}] {info}, target: {self._state.get_goals()}")
        # return info

    def _get_default_state(self, mode: str = "default"):
        """
        Get default values for each JVM option stored in `self._flags`
        from `java -XX:+PrintFlagsFinal -version` command output.
        Also runs a benchmark with default JVM_OPTS and get the initial
        goal value.

        Parameters:
        mode (str): The mode of generating JVM options (default/random)
                    where
                    - "default" sets JVM_OPTS default JDK options,
                    - "random" sets JVM_OPTS random values within options
                    range.

        Returns:
        (np.array) The initial state of the Agent which stores the default
        JVM_OPTS and its performance measurement.
        """
        assert mode in ["default", "random"], f"Unknown mode: {mode}"

        flag_values = []

        if mode == "default":
            flag_values = [7, 12, 4]
            assert len(flag_values) == self._num_flags, "Fix default flag values"
        if mode == "random":
            for i in range(self._num_flags):

                min_val = self._flags_min_values[i]
                max_val = self._flags_max_values[i]
                step_val = self._flags_step_values[i]

                flag_values.append(
                    random.randrange(
                        start=min_val,
                        stop=max_val + step_val,  # must be included
                        step=step_val,
                    )
                )

        # TODO: Replace this to self._run_benchmark
        # state = self._synthetic_run(flag_values)
        state = self._run(flag_values)
        return state

    def _run(self, flag_values: List):
        """
        Run the benchmark with default values and
        get a start point of the goal.

        Parameters:
        flag_values (list): An array of JVM options values.

        Returns:
        (int) A performance measurement (target metric
        or goal value).
        """
        # Convert to List of str
        jvm_opts = self._get_jvm_options(flag_values)

        self._run_benchmark(
            jvm_opts,
            self._gc_log_file,
            self.bm_name,
            self.bm_path,
            self.callback_path,
            self._n,
        )

        if os.path.exists(self._gc_log_file):
            # Get goal value from first-time generated GC log
            param_values, goal_values = self._get_metrics_from_file(
                self._params_names, self._goals_names
            )
            # Clean up
            os.remove(self._gc_log_file)

            state = JVMState(flag_values, param_values, goal_values)

            return state
        else:
            raise Exception(
                "GC log file was not generated: please, ensure that benchmark runs correctly!"
            )

    def _clip_state(self, state: JVMState):
        """
        Use `np.clip` to make sure each JVM flag
        does not leave the boundaries.
        """
        flags = np.clip(
            state.get_flags(), self._flags_min_values, self._flags_max_values
        )

        state.set_flags(flags)

        return state

    def _get_metrics_from_file(self, params, goals):
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
            summary_df = pd.read_csv(
                summary_file, sep=sep, skiprows=1, header=None
            ).replace(",", "", regex=True)
            summary_df.iloc[:, 1] = summary_df.iloc[:, 1].replace(
                "n.a.", "NaN", regex=True
            )
            param_values = (
                summary_df.set_index(0).loc[params].iloc[:, 0].values.astype(np.float32)
            )
            goal_values = (
                summary_df.set_index(0).loc[goals].iloc[:, 0].values.astype(np.float32)
            )

        assert len(param_values) != 0, "Params not found"
        assert len(goal_values) != 0, "Goals not found"

        # if os.path.exists(summary):
        #     os.remove(summary)

        return param_values, goal_values

    def _get_reward(
        self,
        current_state: JVMState,
        previous_state: JVMState,
        lower_is_better: List[bool],
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
        # TODO: Check if beta is in range [0, 1]

        reward_in = 0  # Intrinsic reward.
        reward_ex = 0  # Extrinsic reward.

        # JVMState supports multiple goal values,
        # so we select the first parameter from list
        # TODO: #8 Add multi-objective support
        assert len(current_state.get_goals()) == 1, "Multi-objective is not supported"
        current_goal = current_state.get_goals()[0]
        previous_goal = previous_state.get_goals()[0]
        coef = -1 if lower_is_better[0] else 1

        if coef * current_goal <= previous_goal:
            for i in self._perf_states.keys():
                if np.array_equal(
                    self._perf_states[i]["flags"], current_state.get_flags()
                ):
                    # First, we add the state to a cache with count=1,
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

    def _get_jvm_options(self, run_flags: List):
        """Return a list of JVM options in the format:
        [-XX:<flag_name>=<flag_value><flag_unit>]

        Returns:
            List: JVM options with values.
        """
        jvm_options = []
        for i, flag_info in enumerate(self._flags):
            flag_name = flag_info["name"]
            flag_value = run_flags[i]
            flag_unit = flag_info["unit"]
            if isinstance(flag_value, bool):
                flag_indicator = "+" if flag_value else "-"
                jvm_options.append(f"-XX:{flag_indicator}{flag_name}")
            else:
                jvm_options.append(f"-XX:{flag_name}={flag_value}{flag_unit}")

        return jvm_options

    def _run_benchmark(
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
        try:
            os.remove(gc_log_file)
        except FileNotFoundError:
            pass

        # Default flags
        jvm_opts.append(*self._default_flags)
        # jvm_opts.append("-XX:+UseParallelGC")
        # jvm_opts.append("-Xmx16G")
        # jvm_opts.append("-Xms16G")
        # jvm_opts.append("-XX:SurvivorRatio=130")
        # jvm_opts.append("-XX:TargetSurvivorRatio=66")

        # Run the benchmark (hide output)
        try:
            logging.debug("Running benchmark %s with JVM_OPTS: %s", bm, jvm_opts)
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
            f"Benchmark: {self.bm_name} ({self.bm_path}),\n",
            f"Number of iterations: {self._n},\n",
            f"Goals: {self._goals},\n",
            f"Number of JVM options: {self._num_flags},\n",
            f"JVM options: {self._flags},\n",
            # f"Env. default state: {self._default_state},\n",
            # f"Env. default goal value: {self._default_state[self._goal_idx]},\n",
        )
