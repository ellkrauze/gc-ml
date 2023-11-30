import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def get_data_from_csv(csv_dir: str, goal: str):
    """Parse each summary.csv file stored in `csv_dir`.
    Directory `csv_dir` shall contain summary files
    that describe performance metrics for certain 
    JVM flags.

    E.g., summary_4_12.csv describes performance metrics
    (avgGCPause; 0.0014; seconds) for the JVM with 
    flag_1=4 and flag_2=12.
    Args:
        csv_dir (str):  Path to a directory that contains
            summary*.csv files from gcviewer.jar.
        goal (str):     A key name.
    """
    sep = ';'
    flag_1_values = []
    flag_2_values = []
    goal_values = []

    for summary in os.listdir(csv_dir):
        basename = os.path.splitext(summary)[0]
        p, m = basename.split('_')[-2], basename.split('_')[-1]
        summary_abs_path = os.path.join(csv_dir, summary)

        with open(summary_abs_path, "+r") as summary_file:
            for line in summary_file.readlines():
                if goal + sep in line:
                    goal_value = line.split(sep)[1].replace(',','')
                    goal_value = float(goal_value)

        flag_1_values.append(p)
        flag_2_values.append(m)
        goal_values.append(goal_value) 

    return flag_1_values, flag_2_values, goal_values


def read_single_log(key, filename):
  """bench: avg"""
  (name, n, avg) = (None, 0, 0.00)
  with open(filename) as fd:
    for ln in fd:
      if ln.startswith(key):
        f = ln.split(",")
        (name, n, avg) = (f[1], n + 1, avg + float(f[3]))
  avg = avg/n if n > 0 else 0
  return (name, avg)