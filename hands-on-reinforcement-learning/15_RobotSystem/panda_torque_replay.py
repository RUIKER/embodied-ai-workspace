"""Replay a recorded joint1 torque under two damping settings."""
import argparse, csv, json
from pathlib import Path
import mujoco
import numpy as np


def read_source(path):
    with path.open(newline='', encoding='utf-8') as f:
        rows = [r for r in csv.DictReader(f) if r['condition'] == 'baseline_1']
    t = np.array([float(r['control_time_s']) for r in rows])
    torque = np.array([float(r['joint_actuator_torque_Nm']) for r in rows])
    pos = np.array([float(r['position_rad']) for r in rows])
    if len(t) != 400 or not np.allclose(t, np.arange(400) * .002, atol=1e-10, rtol=0):
        raise ValueError('Expected 400 samples at 2 ms intervals')
    if not np.isfinite(np.c_[t, torque, pos]).all():
        raise ValueError('Non-finite data')
    return t, torque, pos


def run(scene, factor, torque):
    m = mujoco.MjModel.from_xml_path(str(scene))
    if not np.isclose(m.opt.timestep, .002, atol=1e-12, rtol=0):
        raise ValueError('Expected 2 ms timestep')
    j, a = m.joint('joint1').id, m.actuator('actuator1').id
    if m.actuator_trnid[a, 0] != j or m.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
        raise ValueError('actuator1 must drive joint1')
    dof, qadr = m.jnt_dofadr[j], m.jnt_qposadr[j]
    damping = float(m.dof_damping[dof]) * factor
    m.dof_damping[dof] = damping
    # Disable only actuator1. Apply its recorded generalized torque directly;
    # all other actuators remain as in the source experiment.
    m.actuator_gainprm[a, :] = 0
    m.actuator_biasprm[a, :] = 0
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key('home').id)
    mujoco.mj_forward(m, d)
    samples = []
    for u in torque:
        d.qfrc_applied[dof] = u
        mujoco.mj_step(m, d)
        samples.append((d.time, float(d.qpos[qadr]), float(d.qvel[dof]), float(d.qfrc_actuator[dof])))
    arr = np.asarray(samples)
    if not np.isfinite(arr).all() or any(w.number for w in d.warning):
        raise RuntimeError('Simulation warning or non-finite output')
    if not np.allclose(arr[:, 3], 0, atol=1e-12):
        raise RuntimeError('joint1 actuator was not disabled')
    return damping, arr


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('scene', type=Path)
    p.add_argument('source_csv', type=Path)
    p.add_argument('output', type=Path, help='New output directory')
    a = p.parse_args()
    t, torque, original = read_source(a.source_csv)
    b0, x0 = run(a.scene, 1., torque)
    b1, x1 = run(a.scene, 2., torque)
    ref_error = float(np.max(np.abs(x0[:, 1] - original)))
    out = a.output
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'replay.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['control_time_s','sample_time_s','replayed_torque_Nm','baseline_position_rad','damped_position_rad','baseline_velocity_rad_s','damped_velocity_rad_s'])
        for i in range(len(t)):
            w.writerow([t[i], x0[i,0], torque[i], x0[i,1], x1[i,1], x0[i,2], x1[i,2]])
    summary = {'scene':str(a.scene.resolve()), 'source_csv':str(a.source_csv.resolve()),
               'mujoco':mujoco.__version__, 'baseline_damping':b0, 'changed_damping':b1,
               'baseline_replay_max_error_rad':ref_error,
               'max_abs_position_difference_rad':float(np.max(np.abs(x1[:,1]-x0[:,1]))),
               'final_position_difference_rad':float(x1[-1,1]-x0[-1,1]),
               'note':'Open-loop baseline may differ from the recorded closed-loop trajectory; do not attribute that difference to damping.',
               'method':'actuator1 disabled; recorded generalized torque applied at joint1; other actuators unchanged'}
    report=json.dumps(summary, indent=2, allow_nan=False)
    (out/'summary.json').write_text(report+'\n', encoding='utf-8')
    print(report)

if __name__ == '__main__':
    main()
