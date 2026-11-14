"""Compare the existing Gym-like environment with the ROS physics loop offline.

This executes the physics loop extracted from the production ROS source, but
does not start a ROS node or test transport, scheduling, callbacks or heartbeat.
"""
import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from panda_two_joint_env import PandaTwoJointEnv


HERE = Path(__file__).resolve().parent
ROS_SOURCE = HERE / 'ros2_ws/src/panda_sim/panda_sim/node.py'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ros_loop_factory():
    """Compile the unmodified for-loop body from the real publish_state callback."""
    source = ROS_SOURCE.read_text(encoding='utf-8')
    tree = ast.parse(source)
    callbacks = [node for node in ast.walk(tree)
                 if isinstance(node, ast.FunctionDef) and node.name == 'publish_state']
    if len(callbacks) != 1:
        raise RuntimeError('ROS publish_state callback changed; review this audit')
    loops = [node for node in callbacks[0].body if isinstance(node, ast.For)
             and isinstance(node.iter, ast.Call)
             and ast.unparse(node.iter) == 'range(steps_per_message)']
    if len(loops) != 1 or not any(
        isinstance(node, ast.Call) and ast.unparse(node.func) == 'mujoco.mj_step'
        for node in ast.walk(loops[0])
    ):
        raise RuntimeError('ROS physics loop changed; review this audit')
    # The source loop references nonlocal active and the same globals as the node.
    loop_source = ast.unparse(loops[0])
    indented = '\n'.join('        ' + line for line in loop_source.splitlines())
    wrapper = (
        'def make(active, data, model, actuators, steps_per_message, node):\n'
        '    def run():\n'
        '        nonlocal active\n' + indented + '\n'
        '    return run\n'
    )
    namespace = {'np': np, 'NANOSECONDS_PER_SECOND': 1_000_000_000}
    exec(compile(wrapper, str(ROS_SOURCE), 'exec'), namespace)
    return namespace['make'], namespace


def capture_step(model, data, actuators, qadr, vadr, records):
    item = {'relative_s': float(data.time), 'ctrl_before': data.ctrl[actuators].copy()}
    mujoco.mj_step(model, data)
    item.update(q_after=data.qpos[qadr].copy(), dq_after=data.qvel[vadr].copy())
    records.append(item)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('output', type=Path, help='new output directory')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env = PandaTwoJointEnv(args.scene)
    observation, _ = env.reset()
    action = np.array([0.02, -0.01])
    env_records = []
    original_step = mujoco.mj_step

    def trace_env(model, data):
        item = {'relative_s': float(data.time), 'ctrl_before': data.ctrl[env.actuators].copy()}
        original_step(model, data)
        item.update(q_after=data.qpos[env.qadr].copy(), dq_after=data.qvel[env.vadr].copy())
        env_records.append(item)

    try:
        mujoco.mj_step = trace_env
        final_obs, reward, terminated, truncated, info = env.step(action)
    finally:
        mujoco.mj_step = original_step

    make, namespace = ros_loop_factory()
    model = env.model
    actuators7 = np.array([model.actuator(f'actuator{i}').id for i in range(1, 8)])
    start = observation[6:8]
    end = info['target_rad']

    def run_ros_loop(ramp):
        data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
        mujoco.mj_forward(model, data)
        start7 = data.ctrl[actuators7].copy()
        end7 = start7.copy()
        end7[:2] = end
        if ramp:
            active = (data.time, np.array([0, 20_000_000]), np.stack([start7, end7]))
        else:
            data.ctrl[actuators7] = end7
            active = None
        records = []
        namespace['mujoco'] = SimpleNamespace(mj_step=lambda m, d: capture_step(
            m, d, env.actuators, env.qadr, env.vadr, records))
        logger = SimpleNamespace(info=lambda message: None)
        node = SimpleNamespace(get_logger=lambda: logger)
        make(active, data, model, actuators7, 10, node)()
        return records

    held = run_ros_loop(False)
    ramp = run_ros_loop(True)
    assert len(env_records) == len(held) == len(ramp) == 10
    def max_diff(other, field):
        return float(max(np.max(np.abs(a[field] - b[field]))
                         for a, b in zip(env_records, other, strict=True)))
    held_diffs = {name: max_diff(held, name) for name in ('ctrl_before', 'q_after', 'dq_after')}
    ramp_diffs = {name: max_diff(ramp, name) for name in ('ctrl_before', 'q_after', 'dq_after')}
    if max(held_diffs.values()) > 1e-12:
        raise AssertionError(f'Fixed-target control mismatch: {held_diffs}')
    expected = np.stack([start + (end - start) * (i / 10) for i in range(10)])
    np.testing.assert_allclose(np.stack([r['ctrl_before'] for r in ramp]), expected,
                               rtol=0, atol=1e-12)
    if ramp_diffs['ctrl_before'] < 1e-6 or ramp_diffs['q_after'] < 1e-9:
        raise AssertionError('Expected interpolation mismatch was not observed')

    linear_env = PandaTwoJointEnv(args.scene, control_mode='linear')
    linear_env.reset()
    linear_records = []
    def trace_linear(model, data):
        item = {'relative_s': float(data.time),
                'ctrl_before': data.ctrl[linear_env.actuators].copy()}
        original_step(model, data)
        item.update(q_after=data.qpos[linear_env.qadr].copy(),
                    dq_after=data.qvel[linear_env.vadr].copy())
        linear_records.append(item)
    try:
        mujoco.mj_step = trace_linear
        linear_obs, _, linear_terminated, linear_truncated, linear_info = linear_env.step(action)
    finally:
        mujoco.mj_step = original_step
    assert len(linear_records) == 10
    linear_diffs = {name: float(max(np.max(np.abs(a[name] - b[name]))
                    for a, b in zip(linear_records, ramp, strict=True)))
                    for name in ('ctrl_before', 'q_after', 'dq_after')}
    if max(linear_diffs.values()) > 1e-12:
        raise AssertionError(f'Linear mode mismatch: {linear_diffs}')
    np.testing.assert_allclose(linear_obs[6:8], linear_info['held_target_rad'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(linear_info['applied_increment_rad'], end * 0.9, atol=1e-12, rtol=0)
    assert not linear_terminated and not linear_truncated

    with (args.output / 'physics_steps.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['mode', 'step', 'start_s', 'ctrl1_before', 'ctrl2_before',
                         'q1_after', 'q2_after', 'dq1_after', 'dq2_after'])
        for mode, records in [('environment_hold', env_records), ('ros_loop_hold', held),
                              ('ros_loop_interpolation', ramp), ('environment_linear', linear_records)]:
            for i, record in enumerate(records, 1):
                writer.writerow([mode, i, record['relative_s'], *record['ctrl_before'],
                                 *record['q_after'], *record['dq_after']])
    summary = {
        'scope': 'offline production ROS physics loop extracted by AST; no ROS transport or callbacks',
        'action_increment_rad': action.tolist(), 'goal_rad': env.goal.tolist(),
        'initial_observation': observation.tolist(), 'environment_final_observation': final_obs.tolist(),
        'environment_reward': reward, 'terminated': terminated, 'truncated': truncated,
        'outcome': info['outcome'], 'physics_timestep_s': float(model.opt.timestep),
        'physical_steps': 10, 'trajectory_duration_s': 0.02,
        'held_target_max_abs_differences': held_diffs,
        'linear_trajectory_max_abs_differences': ramp_diffs,
        'linear_environment_vs_ros_loop_max_abs_differences': linear_diffs,
        'linear_environment_final_observation': linear_obs.tolist(),
        'linear_environment_requested_target_rad': linear_info['target_rad'].tolist(),
        'linear_environment_held_target_rad': linear_info['held_target_rad'].tolist(),
        'last_step_interpolated_ctrl_rad': ramp[-1]['ctrl_before'].tolist(),
        'scene_xml_sha256': digest(args.scene), 'environment_source_sha256': digest(HERE / 'panda_two_joint_env.py'),
        'ros_source_sha256': digest(ROS_SOURCE), 'audit_source_sha256': digest(Path(__file__)),
        'mujoco_version': mujoco.__version__, 'numpy_version': np.__version__,
        'notes': 'Scene XML hash does not cover referenced model assets; evaluate no stochastic seeds.',
    }
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
