import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_dataset(rb, flags=None):
    """Plot the frequency of flag pairs in a dataset.
    Only works for 2D observations (e.g., for 2 flags in JVMEnv)

    Args:
        rb (tf_agents.replay_buffers.tf_uniform_replay_buffer): Replay buffer
            that stores batches of trajectories collected with policy.
        flags (lst of strings): Array of flag names.
    """

    observations = rb.gather_all()[1].numpy()
    rewards = rb.gather_all()[5].numpy()

    if not flags:
        flags = ["MaxTenuringThreshold", "ParallelGCThreads"]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title("Trajectories Frequency")
    ax.invert_yaxis()
    df = pd.DataFrame(observations.squeeze(), columns=flags)
    x = df.pivot_table(index=flags[0], columns=flags[1],aggfunc='size',fill_value=0)
    idx = x.max(axis=1).sort_values(ascending=0).index
    sns.heatmap(x, annot=True, ax=ax, fmt="")
    plt.show()
    return

def plot_goal_heatmap(env, flags=None, goal: str="Average GC Pause"):
    if flags:
        assert len(flags) == 2, "This function supports only a 2D heatmap"
    else:
        flags = ["MaxTenuringThreshold", "ParallelGCThreads"]
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(5, 5))
    
    cached_states = env._perf_states
    X = [cached_states[i]["args"][0] for i in cached_states.keys()]
    Y = [cached_states[i]["args"][1] for i in cached_states.keys()]
    Z = [cached_states[i]["goal"] for i in cached_states.keys()]
    
    data = pd.DataFrame({
        flags[0]: X, 
        flags[1]: Y, 
        goal: Z})
    data_pivoted = data.pivot(index=flags[0], columns=flags[1], values=goal)
    ax = sns.heatmap(data_pivoted, annot=True, ax=ax, fmt=".2f")
    ax.invert_yaxis()
    plt.show()