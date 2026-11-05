"""Gymnasium environment helpers for the PyTorch implementation."""

from typing import Optional, Tuple

import gymnasium as gym
import numpy as np


def make_environment(name: str, seed: Optional[int] = None):
    """Create an environment and seed it when requested."""
    environment = gym.make(name)
    if seed is not None:
        environment.reset(seed=seed)
        environment.action_space.seed(seed)
    return environment


def reset_environment(environment) -> np.ndarray:
    """Return a reset observation as a float32 array."""
    observation, _ = environment.reset()
    return np.asarray(observation, dtype=np.float32)


def step_environment(
    environment, action: np.ndarray
) -> Tuple[np.ndarray, float, bool, bool]:
    """Step the environment and keep terminated/truncated separate.

    ``terminated`` means a true terminal state (goal reached), while
    ``truncated`` only means the time limit was hit. Bootstrapping must
    continue across truncation, so callers need both flags.
    """
    observation, reward, terminated, truncated, _ = environment.step(action)
    return (
        np.asarray(observation, dtype=np.float32),
        float(reward),
        bool(terminated),
        bool(truncated),
    )
