import random

import numpy as np
import torch

import config
from agents import DDPG
from environment import make_environment
from rl_utils import ReplayBuffer
from trainer import train_off_policy_agent
from visualization import plot_training_results


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(config.SEED)
    env, state_dim, action_dim, action_low, action_high = make_environment(
        config.ENV_NAME
    )
    agent = DDPG(
        state_dim, config.HIDDEN_DIM, action_dim, action_low, action_high,
        config.ACTOR_LR, config.CRITIC_LR, config.GAMMA, config.TAU,
        config.NOISE_TYPE, config.NOISE_SIGMA, config.OU_THETA, config.DEVICE,
    )
    replay_buffer = ReplayBuffer(config.BUFFER_SIZE)
    return_list = train_off_policy_agent(
        env, agent, config.NUM_EPISODES, replay_buffer,
        config.MINIMAL_SIZE, config.BATCH_SIZE, seed=config.SEED,
    )
    plot_training_results(return_list, config.ENV_NAME)
    env.close()


if __name__ == '__main__':
    main()
