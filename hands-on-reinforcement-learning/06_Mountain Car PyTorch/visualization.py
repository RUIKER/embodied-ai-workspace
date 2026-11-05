"""Training result visualization."""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from trainer import EpisodeStats


def plot_episode_stats(
    stats: EpisodeStats, smoothing_window: int, output_dir: Optional[Path] = None
) -> None:
    rewards = np.asarray(stats.rewards)
    lengths = np.asarray(stats.lengths)
    if len(rewards) >= smoothing_window:
        smoothed = np.convolve(
            rewards, np.ones(smoothing_window) / smoothing_window, mode="valid"
        )
        offset = smoothing_window - 1
    else:
        smoothed = rewards
        offset = 0

    figure, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(rewards, alpha=0.35, label="raw reward")
    axes[0].plot(
        np.arange(len(smoothed)) + offset, smoothed, label="smoothed reward"
    )
    axes[0].set(xlabel="Episode", ylabel="Reward")
    axes[0].legend()
    axes[1].plot(lengths)
    axes[1].set(xlabel="Episode", ylabel="Steps")
    figure.suptitle("Mountain Car Actor-Critic with PyTorch")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(figure)
