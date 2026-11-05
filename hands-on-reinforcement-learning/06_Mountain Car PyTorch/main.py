"""Train the PyTorch Mountain Car actor-critic."""

from pathlib import Path
import random

import numpy as np
import torch

from config import (
    ACTOR_LEARNING_RATE,
    CRITIC_LEARNING_RATE,
    DISCOUNT_FACTOR,
    ENTROPY_COEFFICIENT,
    ENVIRONMENT_NAME,
    HIDDEN_SIZE,
    NUM_EPISODES,
    N_STEP,
    PRINT_INTERVAL,
    RANDOM_SEED,
    REWARD_SCALE,
    SMOOTHING_WINDOW,
)
from environment import make_environment
from networks import Actor, Critic
from trainer import train
from visualization import plot_episode_stats


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    output_dir = Path(__file__).resolve().parent / "output"
    environment = make_environment(ENVIRONMENT_NAME, RANDOM_SEED)
    try:
        state_size = int(environment.observation_space.shape[0])
        action_size = int(environment.action_space.shape[0])
        actor = Actor(state_size, action_size, HIDDEN_SIZE)
        critic = Critic(state_size, HIDDEN_SIZE)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=ACTOR_LEARNING_RATE)
        critic_optimizer = torch.optim.Adam(critic.parameters(), lr=CRITIC_LEARNING_RATE)
        stats = train(
            environment,
            actor,
            critic,
            actor_optimizer,
            critic_optimizer,
            NUM_EPISODES,
            DISCOUNT_FACTOR,
            ENTROPY_COEFFICIENT,
            N_STEP,
            REWARD_SCALE,
            PRINT_INTERVAL,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(
            output_dir / "episode_stats.csv",
            np.column_stack((stats.rewards, stats.lengths)),
            delimiter=",",
            header="episode_reward,episode_length",
            comments="",
        )
        torch.save(
            {"actor": actor.state_dict(), "critic": critic.state_dict()},
            output_dir / "actor_critic.pt",
        )
        plot_episode_stats(stats, SMOOTHING_WINDOW, output_dir)
    finally:
        environment.close()


if __name__ == "__main__":
    main()
