"""Fixed seven-joint MuJoCo position tracking baseline; no ROS or learning."""
import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('output', type=Path, help='New output directory')
    args = parser.parse_args()
    scene = args.scene.resolve(strict=True)
    model = mujoco.MjModel.from_xml_path(str(scene))
    dt = float(model.opt.timestep)
    if not np.isclose(dt, 0.002, rtol=0, atol=1e-12):
        raise ValueError('This fixed protocol requires a 2 ms physics timestep')
    joints = [model.joint(f'joint{i}').id for i in range(1, 8)]
    acts = [model.actuator(f'actuator{i}').id for i in range(1, 8)]
    for joint, act in zip(joints, acts):
        if model.actuator_trntype[act] != mujoco.mjtTrn.mjTRN_JOINT or model.actuator_trnid[act, 0] != joint:
            raise ValueError('Actuator-to-joint mapping mismatch')
        if not model.jnt_limited[joint] or not model.actuator_ctrllimited[act]:
            raise ValueError('Protocol requires limited joints and actuator controls')
    low = np.maximum(model.jnt_range[joints, 0], model.actuator_ctrlrange[acts, 0])
    high = np.minimum(model.jnt_range[joints, 1], model.actuator_ctrlrange[acts, 1])
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key('home').id)
    mujoco.mj_forward(model, data)
    qadr = model.jnt_qposadr[joints]
    dadr = model.jnt_dofadr[joints]
    home = data.ctrl[acts].copy()
    offset_a = np.array([0.1, -0.1, 0, 0, 0, 0, 0])
    offset_b = np.array([-0.1, 0.1, 0, 0, 0, 0, 0])
    times = np.array([0, 0.8, 1.2, 2.0, 2.4])
    goals = np.array([home, home + offset_a, home + offset_a, home + offset_b, home + offset_b])
    if not np.isfinite(goals).all() or np.any(goals < low) or np.any(goals > high):
        raise ValueError('Fixed target outside the joint/actuator limit intersection')
    rows = []
    for step in range(1200):
        control_time = step * dt
        command = np.array([np.interp(control_time, times, goals[:, i]) for i in range(7)])
        data.ctrl[acts] = command
        mujoco.mj_step(model, data)
        if (step + 1) % 10 == 0:
            rows.append([float(data.time), control_time, *command, *data.qpos[qadr], *data.qvel[dadr]])
    samples = np.asarray(rows)
    if not np.isfinite(samples).all() or any(w.number for w in data.warning):
        raise RuntimeError('Non-finite state or MuJoCo warning')
    target, actual = samples[:, 2:9], samples[:, 9:16]
    error = actual - target
    summary = {
        'protocol': 'fixed 7-axis position targets; no random sampling or seed',
        'scene': str(scene), 'mujoco': mujoco.__version__, 'numpy': np.__version__,
        'physics_timestep_s': dt, 'record_interval_s': 10 * dt,
        'sampling': 'control at control_time_s; q and dq after final 2 ms physics step at sample_time_s',
        'waypoint_times_s': times.tolist(), 'waypoints_rad': goals.tolist(),
        'joint_actuator_intersection_lower_rad': low.tolist(), 'joint_actuator_intersection_upper_rad': high.tolist(),
        'samples': len(rows), 'rmse_rad_each_joint': np.sqrt(np.mean(error**2, axis=0)).tolist(),
        'max_abs_error_rad_each_joint': np.max(np.abs(error), axis=0).tolist(),
        'final_abs_error_rad_each_joint': np.abs(error[-1]).tolist(),
        'joint_limit_violation_sampled': bool(np.any(actual < model.jnt_range[joints, 0] - .001) or np.any(actual > model.jnt_range[joints, 1] + .001)),
    }
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'tracking.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['sample_time_s', 'control_time_s'] + [f'target_j{i}_rad' for i in range(1, 8)] + [f'q_j{i}_rad' for i in range(1, 8)] + [f'dq_j{i}_rad_s' for i in range(1, 8)])
        writer.writerows(rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    # Standalone SVG needs no plotting dependencies. The plot compares the recorded applied target to post-step state.
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="920" height="520" viewBox="0 0 920 520">', '<rect width="920" height="520" fill="white"/>', '<text x="40" y="27" font-family="sans-serif" font-size="18">Fixed position baseline: applied target and actual</text>']
    for i in range(2):
        top = 55 + i * 225
        def poly(y):
            return ' '.join(f'{70 + x/2.4*800:.1f},{top + 160 - (v + .16)/.32*150:.1f}' for x, v in zip(samples[:, 0], y))
        svg.extend([f'<path d="M70 {top} V{top+160} H870" fill="none" stroke="#333"/>', f'<text x="8" y="{top+15}" font-family="sans-serif" font-size="14">joint{i+1} (rad)</text>', f'<text x="20" y="{top+23}" font-family="sans-serif" font-size="11">+0.16</text>', f'<text x="20" y="{top+163}" font-family="sans-serif" font-size="11">-0.16</text>', f'<polyline points="{poly(target[:, i])}" fill="none" stroke="#2878b5" stroke-width="2"/>', f'<polyline points="{poly(actual[:, i])}" fill="none" stroke="#d33" stroke-width="2"/>', f'<text x="70" y="{top+180}" font-family="sans-serif" font-size="12">0 s</text><text x="835" y="{top+180}" font-family="sans-serif" font-size="12">2.4 s</text>'])
    svg.append('<text x="500" y="27" font-family="sans-serif" font-size="13" fill="#2878b5">blue: applied target</text><text x="680" y="27" font-family="sans-serif" font-size="13" fill="#d33">red: actual q</text></svg>')
    (out / 'tracking.svg').write_text('\n'.join(svg), encoding='utf-8')
    print(json.dumps(summary, indent=2, allow_nan=False))
    if summary['joint_limit_violation_sampled']:
        raise SystemExit('Joint limit exceeded at a sampled state')


if __name__ == '__main__':
    main()
