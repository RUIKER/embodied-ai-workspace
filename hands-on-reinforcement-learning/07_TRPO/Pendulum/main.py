import random

import numpy as np
import torch

import config
from agents import TRPOContinuous
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
    agent = TRPOContinuous(
        state_dim,
        action_dim,
        config.HIDDEN_DIM,
        action_low,
        action_high,
        config.LMBDA,
        config.KL_CONSTRAINT,
        config.ALPHA,
        config.CRITIC_LR,
        config.GAMMA,
        config.REWARD_SHIFT,
        config.REWARD_SCALE,
        config.DEVICE,
        config.CRITIC_EPOCHS,
        config.MINIBATCH_SIZE,
    )
    return_list = train_on_policy_agent(
        env, agent, config.NUM_EPISODES, config.EPISODES_PER_UPDATE,
        seed=config.SEED,
    )
    plot_training_results(return_list, config.ENV_NAME)
    env.close()


if __name__ == '__main__':
    main()
