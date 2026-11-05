"""训练结果可视化。"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pathlib import Path
from typing import Optional

from trainer import EpisodeStats


def plot_episode_stats(
    stats: EpisodeStats,
    smoothing_window: int = 10,
    output_dir: Optional[Path] = None,
    show: bool = True,
) -> tuple[Figure, np.ndarray[tuple[int, int], np.dtype[object]]]:
    """绘制奖励曲线和 episode 长度曲线。"""
    rewards = np.asarray(stats.episode_rewards)
    lengths = np.asarray(stats.episode_lengths)
    smoothed = np.convolve(
        rewards, np.ones(smoothing_window) / smoothing_window, mode="valid"
    ) if len(rewards) >= smoothing_window else rewards
    figure, axes = plt.subplots(2, 1, figsize=(10, 7), constrained_layout=True)
    axes[0].plot(rewards, alpha=0.35, label="raw reward")
    axes[0].plot(np.arange(len(smoothed)) + max(0, smoothing_window - 1),
                 smoothed, label="smoothed reward")
    axes[0].set(xlabel="Episode", ylabel="Reward")
    axes[0].legend()
    axes[1].plot(lengths)
    axes[1].set(xlabel="Episode", ylabel="Steps")
    figure.suptitle("Mountain Car Actor-Critic Training")
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        figure.savefig(output_dir / "training_curves.png", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(figure)
    return figure, axes