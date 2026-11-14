"""Exercise EMA probe orchestration with fake environment and fake optimizer only.

No MuJoCo simulation, real agent, optimizer update, model fitting, or test-goal
evaluation takes place. All temporary data is synthetic and auto-cleaned.
"""
import importlib.util
import json
from pathlib import Path
import random
import sys
import tempfile
from types import ModuleType, SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
spec = importlib.util.spec_from_file_location('ema_probe_test', HERE / 'runner_ema_probe.py')
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def check_math():
    assert np.allclose(probe.smooth_behavior(np.array([1., -1.]), np.zeros(2), .8), [.8, -.8])
    assert np.allclose(probe.smooth_behavior(np.array([1., -1.]), np.zeros(2), .5), [.5, -.5])
    assert np.allclose(probe.smooth_behavior(np.array([1., -1.]), np.array([.6, .7]), 1.), [1., -1.])
    w = probe.empty_behavior_window()
    before = np.array([0., 0., 0., 0., .1, 0., 0., 0.])
    after = before.copy(); after[0] = .01
    raw, executed = np.array([.4, -.2]), np.array([.32, -.16])
    probe.accumulate_behavior(w, before, after, raw, executed, .02*executed, None, None, 'running')
    assert w['actor_steps'] == 1 and w['projection_count'] == 1 and w['adjacent_actor_pairs'] == 0
    assert np.isclose(w['q_goal_euclidean_progress_sum_rad'], .01)
    assert np.isclose(w['goal_q_max_error_sum_rad'], .1)
    assert np.isclose(w['goal_ctrl_action_projection_sum'], .32)
    probe.accumulate_behavior(w, before, after, raw, executed, .02*executed, raw, executed, 'success')
    assert w['adjacent_actor_pairs'] == 1 and np.isclose(w['raw_pair_dot_sum'], .2)
    assert w['actor_successes'] == 1 and sum(w['q_grid_counts'].values()) == 2
    assert np.allclose(w['raw_pair_previous_sum'], raw)
    probe.accumulate_behavior(w, before, after, -raw, -executed, -.02*executed, raw, executed, 'running')
    assert w['adjacent_actor_pairs'] == 2
    assert np.isclose(w['raw_action_delta_norm_sum'], 2*np.linalg.norm(raw))
    assert np.isclose(w['executed_action_delta_norm_sum'], 2*np.linalg.norm(executed))
    assert probe.classify_outcome(True, False,
        {'physics_warning': False, 'outcome': 'failure', 'actual_limit_violation': True}, True)
    assert probe.training_base_reward(-1., True) == -250.
    shaped = probe.shaped_rewards(before.reshape(1, -1), np.array([-1.]), after.reshape(1, -1),
                                  np.array([False]), .99)
    assert np.isclose(shaped[0], -1 - .99*.09 + .1)


def check_source_goals_and_rng():
    # Uses only synthetic arrays and the real replay algorithm; no simulation.
    sys.path.insert(0, str(PROJECT))
    from panda_her import PandaHERReplayBuffer
    def buffer():
        b = PandaHERReplayBuffer(100, np.array([-1., -1.]), np.array([1., 1.]), 2027)
        for episode in range(2):
            for step in range(10):
                before = np.array([step*.005, 0., 0., 0., .14, .14, 0., 0.])
                after = before.copy(); after[0] += .005
                b.add(before, np.zeros(2), -1., after, step == 9, truncated=False, failed=False)
        return b
    old, new = buffer(), buffer()
    random.seed(9173)
    expected, expected_counts = old.sample(16)
    expected_random = random.getstate(); expected_goal_rng = old.goal_rng.getstate()
    random.seed(9173)
    observed, counts, flags, source_goals = probe.sample_her_with_flags(new, 16, capture_source_goals=True)
    assert counts == expected_counts and random.getstate() == expected_random
    assert new.goal_rng.getstate() == expected_goal_rng
    for a, b in zip(expected, observed): np.testing.assert_array_equal(a, b)
    failed = PandaHERReplayBuffer(4, np.array([-1., -1.]), np.array([1., 1.]), 19)
    before = np.array([0., 0., 0., 0., .14, .14, 0., 0.])
    after = before.copy(); after[0] = 1.01
    failed.add(before, np.zeros(2), -250., after, True, truncated=False, failed=True)
    transition, label = failed._relabel(failed.buffer[0])
    assert label == 'source_failed' and transition[2] == -250. and transition[-1] is True
    np.testing.assert_array_equal(transition[0][4:6], [.14, .14])
    np.testing.assert_array_equal(source_goals, np.tile([.14, .14], (16, 1)))
    assert any(flags)
    for i, flag in enumerate(flags):
        assert bool(flag) == (not np.array_equal(observed[0][i, 4:6], source_goals[i]))
    old, new = buffer(), buffer()
    random.seed(1974)
    expected, expected_counts = old.sample(16)
    expected_random, expected_goal_rng = random.getstate(), old.goal_rng.getstate()
    random.seed(1974)
    observed, counts, _, source_goals = probe.sample_her_with_flags(new, 16, capture_source_goals=False)
    assert source_goals is None and counts == expected_counts and random.getstate() == expected_random
    assert new.goal_rng.getstate() == expected_goal_rng
    for a, b in zip(expected, observed): np.testing.assert_array_equal(a, b)


def check_fake_orchestration():
    instances = []
    class FakeEnvironment:
        def __init__(self, scene, max_steps):
            self.lower = self.actual_lower = np.array([-1., -1.])
            self.upper = self.actual_upper = np.array([1., 1.])
            self.data = SimpleNamespace(qpos=np.zeros(2), qvel=np.zeros(2))
            self.actions = []; instances.append(self)
        def reset(self, goal):
            self.steps = 0
            self.goal = np.array(goal)
            self.obs = np.array([0., 0., 0., 0., .1, .1, 0., 0.])
            return self.obs.copy(), {}
        def step(self, increment):
            self.actions.append(np.array(increment))
            self.steps += 1
            self.obs[6:8] += increment
            self.obs[:2] += .1*increment
            self.data.qpos = self.obs[:2].copy()
            truncated = self.steps >= 250
            return self.obs.copy(), -1., False, truncated, {
                'outcome': 'timeout' if truncated else 'running', 'physics_warning': False,
                'actual_limit_violation': False, 'applied_increment_rad': np.array(increment)}
    class FakeReplay:
        def __init__(self, *args): pass
        def add(self, *args, **kwargs): pass
    class FakeNetwork:
        def parameters(self): return []
    class FakeAgent:
        def __init__(self, **config):
            for name in ('actor', 'critic_1', 'critic_2', 'target_critic_1', 'target_critic_2'):
                setattr(self, name, FakeNetwork())
        def take_action(self, obs): return np.array([.4, -.2])
        def update(self, sample, diagnostics=False): return {}
    fake_env = ModuleType('panda_two_joint_env'); fake_env.PandaTwoJointEnv = FakeEnvironment
    fake_her = ModuleType('panda_her'); fake_her.PandaHERReplayBuffer = FakeReplay
    fake_sac = ModuleType('panda_sac_smoke')
    fake_sac.CONFIG = {'gamma': .99}; fake_sac.ReplayBuffer = FakeReplay; fake_sac.SAC = FakeAgent
    fake_sac.SAC_DIR = PROJECT.parent / '11_SAC' / 'Pendulum'; fake_sac.checkpoint = lambda agent: {}
    fake_orig = ModuleType('panda_reach'); fake_orig.TRAIN_GOALS = [(0., 0.)]
    fake_orig.normalized_action = lambda x: np.asarray(x)
    fake_orig.ACTION_SCALE = .02; fake_orig.vector = lambda x: np.asarray(x).tolist()
    fake_orig.model_info = lambda scene: {}
    for name, module in [('panda_two_joint_env', fake_env), ('panda_her', fake_her),
                         ('panda_sac_smoke', fake_sac), ('panda_reach', fake_orig)]:
        sys.modules[name] = module
    probe.verify_environment = lambda p, scene: None
    probe.check_device = lambda d: None
    fake_state = np.tile(np.array([0., 0., 0., 0., .1, .1, 0., 0.]), (64, 1))
    fake_sample = (fake_state, np.zeros((64, 2)), -np.ones(64), fake_state.copy(), np.zeros(64, dtype=bool))
    probe.sample_her_with_flags = lambda replay, size, capture_source_goals=False: (
        fake_sample, {'relabeled': 0}, [False]*size,
        fake_state[:, 4:6].copy() if capture_source_goals else None)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        scene = root / 'scene.xml'; scene.write_text('<fake/>'); (root / 'panda.xml').write_text('<fake/>')
        protocol = {'training_device': 'cpu', 'arms': probe.ARMS,
                    'behavior_coefficients': probe.BEHAVIOR_COEFFICIENTS,
                    'steps': 320, 'warmup': 256, 'batch': 64, 'milestones': [320]}
        (root/'protocol.json').write_text(json.dumps(protocol), encoding='utf-8')
        for arm in probe.ARMS:
            probe.train(root, 1301, arm, scene)
            result = json.loads((root/'seed-1301'/arm/'train-summary.json').read_text(encoding='utf-8'))
            window = json.loads((root/'seed-1301'/arm/'behavior-windows.jsonl').read_text(encoding='utf-8'))
            update = json.loads((root/'seed-1301'/arm/'updates.jsonl').read_text(encoding='utf-8'))
            assert result['updates'] == 64 and window['actor_steps'] == 64
            assert len(update['sample_source_goals']) == len(update['sample_goals']) == len(update['sample_q']) == len(update['sample_next_q']) == len(update['sample_dones_after_relabel']) == 64
            np.testing.assert_allclose(update['sample_source_goals'], update['sample_goals'])
        first_actor = [env.actions[256] for env in instances]
        np.testing.assert_allclose(first_actor[0], .02*np.array([.4, -.2]))
        np.testing.assert_allclose(first_actor[1], .02*.8*np.array([.4, -.2]))
        np.testing.assert_allclose(first_actor[2], .02*.5*np.array([.4, -.2]))
        np.testing.assert_allclose(instances[0].actions[:256], instances[1].actions[:256])
        np.testing.assert_allclose(instances[0].actions[:256], instances[2].actions[:256])


if __name__ == '__main__':
    check_math()
    check_source_goals_and_rng()
    check_fake_orchestration()
    print('PASS: EMA, diagnostics, HER source-goal matching/RNG parity, and 3-arm fake orchestration; no training')
