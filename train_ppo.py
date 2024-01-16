import json
import logging
import os
from random import randrange

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tf_agents import trajectories
from tf_agents.agents import PPOAgent
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.environments import BatchedPyEnvironment, tf_py_environment
from tf_agents.networks.actor_distribution_network import \
    ActorDistributionNetwork
from tf_agents.networks.value_network import ValueNetwork
from tf_agents.policies import policy_saver, random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.specs import from_spec
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tqdm import tqdm

import wandb
from env.PyEnvironmentsTest import JVMEnv  # !!!

SEED = 42

PROJECT_NAME = "Reinforcement Learning (DQN) - JVM-GC"
WANDB_KEY = "4b077df3688052b0f43705d6b4d712c05fb979b7"

logging.basicConfig(
    filename="gc-ml-ppo.log",
    filemode="a",
    format="%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


tf.random.set_seed(SEED)
np.random.seed(SEED)


def collect_step(environment, policy, replay_buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    return trajectory.from_transition(time_step, action_step, next_time_step)


def save_rb(replay_buffer, path):
    tf.train.Checkpoint(rb=replay_buffer).save(path)


def restore_rb(replay_buffer, path):
    tf.train.Checkpoint(rb=replay_buffer).restore(path)


def get_env_state(environment):
    return environment.current_time_step().observation.numpy().squeeze()[:3]


def compute_avg_return_episodic(
    environment,
    policy,
    num_episodes: int = 10,
    patience: int = 100,
    print_info: bool = False,
):
    """
    Computes the average return of a policy,
    given the policy, environment, and a number of episodes.

    Note: for episodic tasks.
    """
    total_return = 0.0
    environment.reset()

    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        i = 0
        while not time_step.is_last():
            if i >= patience:
                break
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
            i += 1
        total_return += episode_return / i

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def create_networks(observation_spec, action_spec, fc_layer_params):
    actor_net = ActorDistributionNetwork(
        observation_spec,  # input
        action_spec,  # output
        fc_layer_params=fc_layer_params,
        activation_fn=tf.keras.activations.tanh,
        seed=SEED,
    )

    value_net = ValueNetwork(
        observation_spec,  # input
        fc_layer_params=fc_layer_params,
        activation_fn=tf.keras.activations.tanh,
    )

    return actor_net, value_net


def get_cd_and_rb(_env, _agent, size, n_steps, batch_size):
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=_agent.collect_data_spec,  # agent.collect_data_spec
        batch_size=batch_size,  # train_env.batch_size
        max_length=size,  # capacity
    )

    replay_buffer_observer = replay_buffer.add_batch

    collect_driver = DynamicEpisodeDriver(
        _env,
        _agent.collect_policy,
        observers=[replay_buffer_observer],  # + train_metrics,
        num_episodes=n_steps,
    )

    return collect_driver, replay_buffer


def get_tf_env(name, args):
    env = JVMEnv(bm_name=name, **args)
    tf_env = tf_py_environment.TFPyEnvironment(env, isolation=True)
    return tf_env


def train(
    _agent,
    _env_train,
    _env_val,
    collect_driver,
    replay_buffer,
    steps: int = 5000,
    use_wandb: bool = False,
    eval_interval: int = 100,
    batch_size: int = 32,
):
    """
    Train reinforcement learning agent and evaluate
    performance on a separate environment.
    """

    _env_train.reset()
    _env_val.reset()
    _agent.train_step_counter.assign(0)
    _agent.train = common.function(_agent.train)
    time_step = None
    policy_state = _agent.collect_policy.get_initial_state(batch_size)

    loss = []
    observations = []
    rewards = []

    avg_reward = compute_avg_return_episodic(_env_val, _agent.policy, num_episodes=20)
    rewards.append(avg_reward)
    # wandb logger for tuning hyperparameters
    if use_wandb:
        wandb.log({"reward": avg_reward})

    for step in tqdm(range(steps)):
        time_step, policy_state = collect_driver.run(
            time_step=time_step,
            policy_state=policy_state,
            maximum_iterations=50,
        )

        experience = replay_buffer.gather_all()
        train_loss = _agent.train(experience)
        replay_buffer.clear()

        if step % eval_interval == 0:
            avg_reward = compute_avg_return_episodic(
                _env_val, _agent.policy, num_episodes=20
            )

            loss.append(train_loss.loss.numpy())
            # observations.append(obs)
            rewards.append(avg_reward)

            # wandb logger for tuning hyperparameters
            if use_wandb:
                wandb.log({"loss": train_loss.loss, "reward": avg_reward})

            print(f"step = {step}: loss = {train_loss.loss}, reward = {avg_reward}")
    return loss, observations, rewards


def train_random(
    policy,
    _env_train,
    _env_val,
    steps: int = 5000,
    use_wandb: bool = False,
    eval_interval: int = 100,
):
    """
    Get performance metrics on a random policy.
    """

    _env_train.reset()
    _env_val.reset()

    loss = []
    observations = []
    rewards = []

    avg_reward = compute_avg_return_episodic(_env_val, policy, num_episodes=20)
    rewards.append(avg_reward)
    # wandb logger for tuning hyperparameters
    if use_wandb:
        wandb.log({"reward": avg_reward})

    for step in tqdm(range(steps)):
        if step % eval_interval == 0:
            avg_reward = compute_avg_return_episodic(_env_val, policy, num_episodes=20)

            loss.append(0)
            # observations.append(obs)
            rewards.append(avg_reward)

            # wandb logger for tuning hyperparameters
            if use_wandb:
                wandb.log({"loss": 0, "reward": avg_reward})

            print(f"step = {step}: loss = {0}, reward = {avg_reward}")
    return loss, observations, rewards


def main():
    with open("experiment.json", "r", encoding="utf-8") as c:
        conf = json.load(c)

    wandb.init(
        config=dict(
            competition=PROJECT_NAME,
            _wandb_kernel="lemon",
            seed=SEED,
        )
    )

    env_args = {
        "jdk_path": conf["jdk_path"],
        "bm_path": conf["bm_path"],
        "gc_viewer_jar": conf["gc_viewer_jar"],
        "callback_path": conf["callback_path"],
        "n": conf["n"],
        "goal": conf["goal"],
        "verbose": conf["verbose"],
    }

    # Batch together multiple py environments and act as a single batch.
    names = conf["train_benchmarks"]
    test_name = conf["test_benchmark"]
    num_steps = conf["num_steps"]
    dataset_size = conf["dataset_size"]
    fc_layer_params = tuple(conf["fc_layer_params"])
    entropy_regularization = conf["entropy_regularization"]
    importance_ratio_clipping = conf["importance_ratio_clipping"]
    policy_l2_reg = conf["policy_l2_reg"]
    # batch_size = conf["batch_size"]
    learning_rate = conf["learning_rate"]
    eval_interval = conf["eval_interval"]
    train_episodes_per_iteration = conf["train_episodes_per_iteration"]
    policy_dir = os.path.join(conf["policy_dir"])

    envs = [JVMEnv(bm_name=name, **env_args) for name in names]
    batched_env = BatchedPyEnvironment(envs)
    train_env = tf_py_environment.TFPyEnvironment(batched_env)
    test_env = get_tf_env(name=test_name, args=env_args)

    batch_size = train_env.batch_size
    action_spec = from_spec(train_env.action_spec())
    observation_spec = from_spec(train_env.observation_spec())
    reward_spec = from_spec(train_env.reward_spec())
    time_step_spec = trajectories.time_step_spec(observation_spec, reward_spec)

    actor_net, value_net = create_networks(
        observation_spec, action_spec, fc_layer_params
    )
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    global_step = tf.compat.v1.train.get_or_create_global_step()

    agent_args = {
        "optimizer": optimizer,
        "actor_net": actor_net,
        "value_net": value_net,
        "train_step_counter": global_step,
        "entropy_regularization": entropy_regularization,
        "importance_ratio_clipping": importance_ratio_clipping,
        "policy_l2_reg": policy_l2_reg,
    }

    agent = PPOAgent(time_step_spec, action_spec, **agent_args)
    agent.initialize()
    tf_policy_saver = policy_saver.PolicySaver(agent.policy)

    random_policy = random_tf_policy.RandomTFPolicy(
        time_step_spec=time_step_spec,
        action_spec=action_spec,
        automatic_state_reset=True,
        clip=False,
        emit_log_probability=True,
    )
    collect_driver, replay_buffer = get_cd_and_rb(
        train_env, agent, dataset_size, train_episodes_per_iteration, batch_size
    )

    wandb.log(conf)

    train(
        agent,
        train_env,
        test_env,
        collect_driver,
        replay_buffer,
        steps=num_steps,
        use_wandb=True,
        eval_interval=eval_interval,
        batch_size=batch_size,
    )

    # train_random(
    #     random_policy,
    #     train_env,
    #     test_env,
    #     num_steps,
    #     use_wandb=True,
    #     eval_interval=eval_interval,
    # )

    tf_policy_saver.save(policy_dir)


if __name__ == "__main__":
    main()
