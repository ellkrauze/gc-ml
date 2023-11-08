import os
import re
import sys
import gym
import random
import requests
import json
import subprocess
import logging
import numpy as np
from typing import List
from gym.spaces import Discrete, Box


class JVMEnv(gym.Env):
    
    def __init__(
        self,
        jdk: str,
        bm_path: str,
        jvm_opt_name: str = "MaxHeapSize",
        jvm_opt_range: List[np.array] = [np.array([268435456]), np.array([4294967296])],
        bm: str = "cassandra",
        n: int = 5,
        goal: str = "fullGCMaxTime",
        api_key: str = "af176fbc-7be2-4fc5-8b31-5624956845c6"
    ):
        """
        OpenAI Custom Environment for tuning Parallel Garbage Collector.
        This class inferits from the abstract class gym.Env.

        This implementation runs DaCapo benchmark on the selected OpenJDK,
        measures performance, and tunes ParallelGC using Reinforcement
        Learning.

        Parameters:
        jdk (str):          OpenJDK used for benchmarking.
        bm_path (str):      Full path to the benchmark jar.
        jvm_opt_name (str): JVM option to perform action to (increase, leave,
                            decrease). For instance, increase the MaxHeapSize.
        jvm_opt_range (list[np.array]): A numerical range for the specified
                            JVM option to configure.
        bm (str):           Name of the DaCapo benchmark.
        n (int):            Number of benchmark's iterations in total.
        goal (str):         What to optimize during the performance tuning. This is
                            a JSON field which is extracted from GC log file using
                            GCEasy API.
        api_key (str):      GCEasy API key.

        """
        self.env = os.environ.copy()
        self.env["PATH"] = f"{jdk}/bin:{self.env['PATH']}"

        # Actions we can take: down, stay, up
        # {-1, 0, 1}
        self.action_space = Discrete(3, start=-1)
        self.observation_space = Box(low=jvm_opt_range[0], high=jvm_opt_range[1])

        self.jvm_opt_name = jvm_opt_name
        self.jvm_opt_value = self._get_JVM_opt_value(self.jvm_opt_name)
        self.bm = bm
        self.bm_path = bm_path
        self.gc_log_file = f"gc-{self.bm}.txt"
        self.n = n
        self.goal = goal
        self.api_key = api_key

        # This will take a while
        self.default_goal_value = self._get_initial_goal_value()
        self.best_goal_value = self.default_goal_value
        self.current_goal_value = 0

    def reset(self, seed=None, options=None):
        # Reset JVM option to default value
        self.jvm_opt_value = self._get_JVM_opt_value(self.jvm_opt_name)
        self.best_goal_value = self.default_goal_value
        return self.jvm_opt_value

    def step(self, action):
        coef = 100 * (2**10) * (2**10) # 100 Mb

        # Apply action (increase, leave, or decrease memory by 100 Kb)
        # -1*100Kb = -100Kb
        # 0*100Kb = 0
        # 1*100Kb = +100Kb
        self.jvm_opt_value += action * coef

        # 1. Launch JVM with new options
        jvm_opts = self._update_jvm_opts()
        # print(f"Running with {jvm_opts}")
        self._run(jvm_opts, self.gc_log_file, self.bm, self.bm_path, self.n)

        # 2. Measure goal
        self.current_goal_value = self._get_goal_value(self.goal, self.gc_log_file, self.api_key)

        # 3. Compare goal with the start point
        if self.current_goal_value <= self.best_goal_value:
            reward = 1
            # Save the best result
            self.best_goal_value = self.current_goal_value
        else:
            reward = -1

        # TODO: Done when ...?
        if (self.jvm_opt_value <= self.observation_space.low[0] or
            self.jvm_opt_value >= self.observation_space.high[0] or
            self.current_goal_value <= self.default_goal_value // 2):
            done = True
        else:
            done = False

        info = {
            f"current {self.jvm_opt_name}": self.jvm_opt_value,
            "best_goal_value": self.best_goal_value,
            "current_goal_value": self.current_goal_value
        }

        print(info)

        return self._get_obs(), reward, done, info

    def render(self):
        pass

    def close(self):
        pass

    def _update_jvm_opts(self):
        return [f"-XX:{self.jvm_opt_name}={self.jvm_opt_value}"]

    def _get_initial_goal_value(self):
        """
        Run the benchmark with default values
        and get a start point of the goal.
        """
        # Run benchmark with default values
        self._run([], self.gc_log_file, self.bm, self.bm_path, self.n)

        if os.path.exists(self.gc_log_file):
            # Get goal value from first-time generated GC log
            result = self._get_goal_value(self.goal, self.gc_log_file, self.api_key)
            # Clean up
            os.remove(self.gc_log_file)
            return result
        else:
            raise Exception("GC log file was not generated: please, ensure that benchmark runs correctly!")

    # def _get_info(self):
    #   return {
    #       self.jvm_opt_name: self.jvm_opt_value,
    #       "default_goal_value": self.default_goal_value,
    #       "current_goal_value": self.current_goal_value
    #   }

    def _get_obs(self):
        return self.jvm_opt_value

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
            env=self.env)

        grep_process = subprocess.Popen(
            ["grep", opt, "-m", "1"],
            stdin=flags_process.stdout,
            stdout=subprocess.PIPE,
            text=True)

        output, error = grep_process.communicate()
        result = re.findall("\d+", output)[0]

        return int(result)


    def _get_goal_value(self, goal_name: str, gc_log_file: str, api_key: str, verbose: bool=False):
        """
        Use GCEasy API to parse GC log file
        and get the values of parameters to tune:

        curl -X POST --data-binary @./{gc_log_file} \
        https://api.gceasy.io/analyzeGC?apiKey={api_key} \
        --header "Content-Type:text"

        Parameters:
        goal_name (str):   What to optimize in performance tuning.
                        This is a JSON field which is extracted
                        from GC log file using GCEasy API.
                        (minorGCMaxTime / fullGCMaxTime)
        gc_log_file (str): Name of the output file (Garbage Collector
                        logs).
        api_key (str):     GCEasy API Key.
        verbose (bool):    Print debug messages.

        Returns:
        int: Goal value in milliseconds
        """
        gceasy_url = f"https://api.gceasy.io/analyzeGC?apiKey={api_key}&normalizeUnits=true"
        gc_logfile = {"upload_file": open(gc_log_file, 'rb')}
        headers = {"Content-Type": "text"}

        response = requests.post(gceasy_url, files=gc_logfile, headers=headers)
        gc_info = json.loads(response.text)
        print(gc_info)
        goal_value = gc_info["gcStatistics"][goal_name]

        if verbose: logging.debug(f"{goal_name} = {goal_value}")

        return float(goal_value)

    def _run(self, jvm_opts: List[str], gc_log_file: str, bm: str, bm_path: str, n: int=5, verbose: bool=False):
        """
        Run a DaCapo benchmark with the specified JVM options
        such as MaxHeapSize

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

        # Run the benchmark (hide output)
        java_process = subprocess.call(
            ["java",
            "-cp", bm_path,
            f"-Xlog:gc=debug:file={gc_log_file}",  *jvm_opts,
            "-Dvmstat.enable_jfr=yes",
            "-Dvmstat.csv=yes", "Harness", "-v", "-n", f"{n}", bm],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            text = True,
            env = self.env)
        # ! java -cp {bm_path} -Xlog:gc=debug:file={gc_log_file}
        return
