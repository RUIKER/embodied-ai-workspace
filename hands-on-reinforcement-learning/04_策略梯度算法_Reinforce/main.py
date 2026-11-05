import numpy as np
import random
import torch
import config
from environment import make_environment
from visualization import plot_returns
from trainer import train_on_policy_agent
from agents import REINFORCE

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def main():
    set_seed(config.SEED)
    env, state_dim, action_dim = make_environment(config.ENV_NAME)
    agent = REINFORCE(
        state_dim,
        config.HIDDEN_DIM,
        action_dim,
        config.LR,
        config.GAMMA,
        config.DEVICE
    )

    return_list = train_on_policy_agent(env, agent, config.NUM_EPISODES)
    plot_returns(return_list)
    env.close()

if __name__ == '__main__':
    main()