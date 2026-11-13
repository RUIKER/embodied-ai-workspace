"""Read-only and synthetic checks; never steps the simulator or updates an agent."""
import sys
import copy
import random
from pathlib import Path

import numpy as np
import runner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from panda_her import PandaHERReplayBuffer


def main():
    protocol = runner.protocol()
    assert len(protocol['validation_cases']) == len(protocol['test_cases']) == 48
    assert protocol['seeds'] == [1103, 1129, 1181, 1213, 1277]
    runner.environment_snapshot(runner.SCENE)
    raw1 = np.array([1., -.5])
    raw2 = np.array([-.5, 1.])
    executed1 = runner.smooth_behavior(raw1, np.zeros(2))
    executed2 = runner.smooth_behavior(raw2, executed1)
    np.testing.assert_allclose(executed1, [.5, -.25])
    np.testing.assert_allclose(executed2, [0., .375])
    np.testing.assert_allclose(runner.smooth_behavior(raw1, np.zeros(2)), executed1)
    assert np.max(np.abs(executed2)) <= 1.
    gamma = .99
    state = np.array([[0., 0., 0., 0., .1, .1, 0., 0.],
                      [0., 0., 0., 0., .1, .1, 0., 0.]])
    nxt = np.array([[.05, .05, 0., 0., .1, .1, .05, .05],
                    [.05, .05, 0., 0., .1, .1, .05, .05]])
    r = runner.shaped_rewards(state, np.array([-1., -1.]), nxt, np.array([False, True]), gamma)
    np.testing.assert_allclose(r, [-1. + .1 - .99*.05, -.9])
    # Synthetic data, no environment step or neural network operation.
    replay = PandaHERReplayBuffer(128, np.array([-1., -1.]), np.array([1., 1.]), 7)
    for i in range(64):
        s = state[0].copy()
        n = nxt[0].copy()
        replay.add(s, np.array([.3, -.2]), -1., n, False,
                   truncated=(i == 63), failed=False)
    replay_copy = copy.deepcopy(replay)
    random.seed(101)
    sampled, counts = replay.sample(64)
    random.seed(101)
    sampled2, counts2, flags = runner.sample_her_with_flags(replay_copy, 64)
    assert counts2 == counts and sum(flags) == counts['relabeled']
    for a, b in zip(sampled, sampled2):
        np.testing.assert_array_equal(a, b)
    s, a, rewards, ns, dones = sampled
    assert counts['relabeled'] > 0
    np.testing.assert_allclose(s[:, 4:6], ns[:, 4:6])
    shaped = runner.shaped_rewards(s, rewards, ns, dones, gamma)
    assert shaped.shape == (64,) and np.isfinite(shaped).all()
    np.testing.assert_allclose(a, np.tile([.3, -.2], (64, 1)))
    print('PASS: goals, source snapshot, shaped terminal/continuing targets, synthetic HER consistency; no training or simulation')


if __name__ == '__main__':
    main()
