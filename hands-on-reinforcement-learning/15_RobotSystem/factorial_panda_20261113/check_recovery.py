"""Synthetic amended failure-policy checks, without simulator steps or training."""
import importlib.util
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
sys.path.insert(0, str(PROJECT))
spec = importlib.util.spec_from_file_location('runner_recovery', HERE / 'runner_recovery.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
from panda_her import PandaHERReplayBuffer


def fail(fn):
    try:
        fn()
    except RuntimeError:
        return
    raise AssertionError('Expected error')


def main():
    good = {'outcome': 'running', 'physics_warning': False, 'actual_limit_violation': False}
    breach = {'outcome': 'failure', 'physics_warning': False, 'actual_limit_violation': True}
    assert not runner.classify_outcome(False, False, good, True)
    assert runner.classify_outcome(True, False, breach, True)
    fail(lambda: runner.classify_outcome(True, False, {**breach, 'physics_warning': True}, True))
    fail(lambda: runner.classify_outcome(True, False, breach, False))
    fail(lambda: runner.classify_outcome(True, False, {**breach, 'actual_limit_violation': False}, True))
    assert runner.training_base_reward(-1, True) == -250
    assert runner.training_base_reward(-1, False) == -1
    assert runner.training_base_reward(0, False) == 0
    original = np.array([0., 0., 0., 0., .1, .1, 0., 0.])
    next_state = np.array([.05, .05, 0., 0., .1, .1, .05, .05])
    action = np.zeros(2)
    replay = PandaHERReplayBuffer(8, np.array([-.2, -.2]), np.array([.2, .2]), 8)
    replay.add(original, action, -250., next_state, True, truncated=False, failed=True)
    t, flag = replay._relabel(replay.buffer[0])
    assert flag == 'source_failed' and t[2] == -250. and t[4] is True
    replay.add(original, action, -1., next_state, False, truncated=True, failed=False)
    t, flag = replay._relabel(replay.buffer[1])
    assert flag == 'relabeled' and t[2] == 0. and t[4] is True
    assert np.allclose(t[0][4:6], [.05, .05]) and np.allclose(t[3][4:6], [.05, .05])
    a = runner.shaped_rewards(original[None,:], np.array([-250.]), next_state[None,:], np.array([True]), .99)
    b = runner.shaped_rewards(original[None,:], np.array([-1.]), next_state[None,:], np.array([False]), .99)
    c = runner.shaped_rewards(t[0][None,:], np.array([t[2]]), t[3][None,:], np.array([t[4]]), .99)
    assert np.allclose(a, [-249.9]) and np.allclose(b, [-.9495]) and np.allclose(c, [.05])
    print('PASS: finite limit failure; equal failure cost; HER non-relabel; goals, termination and truncation shaping')


if __name__ == '__main__':
    main()
