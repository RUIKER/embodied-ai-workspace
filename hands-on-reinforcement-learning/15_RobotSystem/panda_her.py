"""Goal relabeling for the Panda two-joint sparse reaching task."""

from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, field
import random

import numpy as np

from panda_two_joint_env import SPEED_THRESHOLD_RAD_S, goal_reached


@dataclass
class EpisodeGoals:
    steps: list = field(default_factory=list)
    positions: list = field(default_factory=list)


@dataclass(frozen=True)
class Experience:
    transition: tuple
    episode: EpisodeGoals
    step: int
    failed: bool


class PandaHERReplayBuffer:
    """Sample real transitions and attempt to relabel half of each batch."""

    def __init__(self, capacity, lower, upper, seed):
        self.buffer = deque(maxlen=capacity)
        self.lower, self.upper = lower, upper
        self.goal_rng = random.Random(seed)
        self.episode = EpisodeGoals()
        self.episode_step = 0

    def size(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, terminated, *, truncated, failed):
        self.episode_step += 1
        self.buffer.append(Experience(
            (state, action, reward, next_state, terminated), self.episode, self.episode_step, failed
        ))
        q, velocity = next_state[:2], next_state[2:4]
        if (not failed and np.all(np.abs(velocity) < SPEED_THRESHOLD_RAD_S)
                and np.all(q >= self.lower) and np.all(q <= self.upper)):
            self.episode.steps.append(self.episode_step)
            self.episode.positions.append(q.copy())
        if terminated or truncated:
            self.episode = EpisodeGoals()
            self.episode_step = 0

    def _relabel(self, experience):
        if experience.failed:
            return experience.transition, "source_failed"
        episode = experience.episode
        # Inclusive: this transition's next state is the earliest possible future goal.
        first = bisect_left(episode.steps, experience.step)
        if first == len(episode.steps):
            return experience.transition, "no_future_goal"
        goal = episode.positions[self.goal_rng.randrange(first, len(episode.steps))]
        state, action, _, next_state, _ = experience.transition
        if goal_reached(state[:2], state[2:4], goal):
            return experience.transition, "already_at_goal"
        success = goal_reached(next_state[:2], next_state[2:4], goal)
        state, next_state = state.copy(), next_state.copy()
        state[4:6] = next_state[4:6] = goal
        return (state, action, 0.0 if success else -1.0, next_state, success), "relabeled"

    def sample(self, batch_size):
        # Keep the same uniform draw of real transitions as the original replay.
        experiences = random.sample(self.buffer, batch_size)
        attempts = batch_size // 2
        counts = {"attempts": attempts, "relabeled": 0, "relabeled_success": 0,
                  "no_future_goal": 0, "already_at_goal": 0, "source_failed": 0}
        transitions = []
        for i, experience in enumerate(experiences):
            if i < attempts:
                transition, result = self._relabel(experience)
                counts[result] += 1
                if result == "relabeled" and transition[-1]:
                    counts["relabeled_success"] += 1
            else:
                transition = experience.transition
            transitions.append(transition)
        counts["original"] = batch_size - counts["relabeled"]
        return tuple(np.asarray(values) for values in zip(*transitions)), counts
