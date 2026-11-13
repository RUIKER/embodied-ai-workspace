"""Check Panda HER replay semantics with fixed transitions and no optimizer updates."""

from unittest.mock import patch

import numpy as np

from panda_her import PandaHERReplayBuffer
from panda_two_joint_env import SPEED_THRESHOLD_RAD_S


def observation(position, velocity=(0.0, 0.0)):
    return np.array([*position, *velocity, 0.8, 0.8, 0.1, 0.2])


def sample_first(replay, index):
    # Choose the base transition; the buffer still samples the future goal itself.
    others = [i for i in range(replay.size()) if i != index]
    with patch("panda_her.random.sample", return_value=[replay.buffer[index], replay.buffer[others[0]]]):
        (states, actions, rewards, next_states, dones), counts = replay.sample(2)
    return (states[0], actions[0], rewards[0], next_states[0], dones[0]), counts


def check():
    replay = PandaHERReplayBuffer(8, np.array([-1., -1.]), np.array([1., 1.]), seed=44)
    start = observation((0.0, 0.0))
    fast = observation((0.06, 0.07), (SPEED_THRESHOLD_RAD_S, 0.0))
    settled = observation((0.06, 0.07))
    action = np.array([0.5, -0.5])
    replay.add(start, action, -1., fast, False, truncated=False, failed=False)
    replay.add(fast, action, -1., settled, False, truncated=True, failed=False)

    # Exactly at the virtual position but moving at the strict speed threshold.
    (state, sampled_action, reward, next_state, done), counts = sample_first(replay, 0)
    np.testing.assert_array_equal(state[4:6], settled[:2])
    np.testing.assert_array_equal(next_state[4:6], settled[:2])
    np.testing.assert_array_equal(sampled_action, action)
    keep = [0, 1, 2, 3, 6, 7]
    np.testing.assert_array_equal(state[keep], start[keep])
    np.testing.assert_array_equal(next_state[keep], fast[keep])
    np.testing.assert_array_equal(start[4:6], [0.8, 0.8])
    np.testing.assert_array_equal(fast[4:6], [0.8, 0.8])
    assert reward == -1. and not done and counts["relabeled"] == 1

    # Real timeout bootstraps. For the relabeled sample, reaching the new goal
    # on that same step is a true terminal and masks bootstrap.
    (_, _, reward, _, done), counts = sample_first(replay, 1)
    assert reward == 0. and done and counts["relabeled_success"] == 1
    assert replay.buffer[1].transition[2] == -1. and not replay.buffer[1].transition[-1]

    fast_again = observation((0.2, 0.3), (0.2, 0.3))
    failure = observation((0.25, 0.3))
    replay.add(start, action, -1., fast_again, False, truncated=False, failed=False)
    replay.add(fast_again, action, -1., failure, True, truncated=False, failed=True)
    replay.add(start, action, -1., settled, False, truncated=False, failed=False)
    replay.add(settled, action, -1., settled, False, truncated=True, failed=False)
    (_, _, reward, next_state, done), counts = sample_first(replay, 2)
    assert reward == -1. and not done and counts["no_future_goal"] == 1
    np.testing.assert_array_equal(next_state, fast_again)
    (_, _, reward, next_state, done), counts = sample_first(replay, 3)
    assert reward == -1. and done and counts["source_failed"] == 1
    np.testing.assert_array_equal(next_state, failure)
    (_, _, reward, _, done), counts = sample_first(replay, 5)
    assert reward == -1. and not done and counts["already_at_goal"] == 1

    # A low-speed state before the transition and a low-speed position
    # outside the allowable goal range are both ineligible.
    outside = observation((1.1, 0.4))
    replay.add(start, action, -1., settled, False, truncated=False, failed=False)
    replay.add(settled, action, -1., outside, False, truncated=True, failed=False)
    (_, _, reward, _, done), counts = sample_first(replay, 7)
    assert reward == -1. and not done and counts["no_future_goal"] == 1

    replay.add(start, action, -1., settled, False, truncated=False, failed=False)
    assert replay.size() == 8
    (_, _, reward, _, done), counts = sample_first(replay, 6)
    assert reward == -1. and not done and counts["no_future_goal"] == 1
    (_, _, reward, _, done), counts = sample_first(replay, 7)
    assert reward == 0. and done and counts["original"] == 1


if __name__ == "__main__":
    check()
    print("PASS: HER replay, goal bounds, speed, timeout, failure, and episode boundaries")
