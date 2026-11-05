import random

import numpy as np
import torch

import config
import rl_utils
from agents import build_agent
from environment import make_environment
from trainer import train_DQN
from visualization import plot_training_results


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(config.SEED)
    env, state_dim, action_dim = make_environment(config.ENV_NAME, config.ACTION_DIM)
    replay_buffer = rl_utils.ReplayBuffer(config.BUFFER_SIZE)
    # 算法切换只在config.ALGORITHM中手动完成
    agent = build_agent(
        config.ALGORITHM,
        state_dim,
        config.HIDDEN_DIM,
        action_dim,
        config.LR,
        config.GAMMA,
        config.EPSILON,
        config.TARGET_UPDATE,
        config.DEVICE,
    )
    return_list, max_q_value_list = train_DQN(
        agent,
        env,
        config.NUM_EPISODES,
        replay_buffer,
        config.MINIMAL_SIZE,
        config.BATCH_SIZE,
    )
    plot_training_results(return_list, max_q_value_list, config.ALGORITHM, config.ENV_NAME)


if __name__ == '__main__':
    main()