import random

import numpy as np
import torch

import config
from agents import PPOContinuous
from environment import make_environment
from trainer import train_on_policy_agent
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
    env.reset(seed=config.SEED)
    agent = PPOContinuous(
        state_dim,
        config.HIDDEN_DIM,
        action_dim,
        action_low,
        action_high,
        config.ACTOR_LR,
        config.CRITIC_LR,
        config.LMBDA,
        config.EPOCHS,
        config.EPS,
        config.GAMMA,
        config.REWARD_SHIFT,
        config.REWARD_SCALE,
        config.DEVICE,
        config.MINIBATCH_SIZE,
        config.MAX_GRAD_NORM,
    )
    return_list = train_on_policy_agent(
        env, agent, config.NUM_EPISODES, config.EPISODES_PER_UPDATE,
        seed=config.SEED,
    )
    plot_training_results(return_list, config.ENV_NAME)
    env.close()


if __name__ == '__main__':
    main()
