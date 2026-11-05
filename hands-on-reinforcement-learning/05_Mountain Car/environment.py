"""Gymnasium 环境适配层，隔离版本差异和动作形状处理。"""

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box
from typing import Any, Optional, Protocol, cast


class MountainCarEnvironment(Protocol):
    observation_space: Box
    action_space: Box
    spec: Any

    def reset(self, *, seed: Optional[int] = None) -> tuple[np.ndarray, dict]:
        ...

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        ...

    def close(self) -> None:
        ...


def make_environment(name: str, seed: Optional[int] = None) -> MountainCarEnvironment:
    """创建环境并在需要时设置随机种子。"""
    env = gym.make(name)
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
    return cast(MountainCarEnvironment, env)


def reset_environment(env: MountainCarEnvironment) -> np.ndarray:
    """统一返回 reset 后的 observation。"""
    observation, _ = env.reset()
    return np.asarray(observation, dtype=np.float64)


def step_environment(
    env: MountainCarEnvironment, action: np.ndarray
) -> tuple[np.ndarray, float, bool, bool]:
    """统一 Gymnasium 的返回值，并区分 terminated/truncated。

    terminated 是真正的终止（到达终点），truncated 只是时间限制截断，
    截断状态仍有价值，调用方需要区分两者以正确 bootstrap。
    """
    observation, reward, terminated, truncated, _ = env.step(action)
    return np.asarray(observation, dtype=np.float64), float(reward), terminated, truncated