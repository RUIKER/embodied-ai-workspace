"""Compare a fixed Panda trajectory at two joint1 damping values."""

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np


def make_model(scene, factor):
    model = mujoco.MjModel.from_xml_path(str(scene))
    joint = model.joint("joint1").id
    actuator = model.actuator("actuator1").id
    if (model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT
            or model.actuator_trnid[actuator, 0] != joint):
        raise ValueError("actuator1 does not drive joint1")
    if not np.isclose(model.opt.timestep, 0.002, rtol=0, atol=1e-12):
        raise ValueError("This protocol requires a 2 ms model timestep")
    dof = model.jnt_dofadr[joint]
    model.dof_damping[dof] *= factor
    return model, joint, actuator, dof


def run(model, joint, actuator, dof):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    start_target = float(data.ctrl[actuator])
    end_target = start_target + 0.1
    lower = max(model.jnt_range[joint, 0], model.actuator_ctrlrange[actuator, 0])
    upper = min(model.jnt_range[joint, 1], model.actuator_ctrlrange[actuator, 1])
    if not lower <= start_target <= upper or not lower <= end_target <= upper:
        raise ValueError("The fixed trajectory is outside the joint/actuator limits")

    start_time = data.time
    position_address = model.jnt_qposadr[joint]
    rows = []
    for step in range(400):
        control_time = step * model.opt.timestep
        target = start_target + 0.1 * min(control_time / 0.4, 1.0)
        data.ctrl[actuator] = target
        mujoco.mj_step(model, data)
        rows.append([data.time - start_time, control_time, target,
                     data.qpos[position_address], data.qvel[dof],
                     data.actuator_force[actuator], data.qfrc_actuator[dof]])
    samples = np.asarray(rows)
    if not np.isfinite(samples).all() or any(warning.number for warning in data.warning):
        raise RuntimeError("Simulation produced a non-finite value or warning")
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("output", type=Path, help="New output directory")
    parser.add_argument("--damping-factor", type=float, default=2.0)
    args = parser.parse_args()
    if not np.isfinite(args.damping_factor) or args.damping_factor <= 0:
        parser.error("--damping-factor must be finite and positive")

    scene = args.scene.resolve(strict=True)
    baseline, joint, actuator, dof = make_model(scene, 1.0)
    damped, joint_damped, actuator_damped, dof_damped = make_model(scene, args.damping_factor)
    original_damping = float(baseline.dof_damping[dof])
    changed_damping = float(damped.dof_damping[dof_damped])
    samples = {
        "baseline_1": run(baseline, joint, actuator, dof),
        "baseline_2": run(baseline, joint, actuator, dof),
        "damped_1": run(damped, joint_damped, actuator_damped, dof_damped),
        "damped_2": run(damped, joint_damped, actuator_damped, dof_damped),
    }
    reference_time = samples["baseline_1"][:, 0]
    for values in samples.values():
        np.testing.assert_allclose(values[:, 0], reference_time, rtol=0, atol=1e-12)
        np.testing.assert_allclose(values[:, 2], samples["baseline_1"][:, 2], rtol=0, atol=1e-12)

    original = samples["baseline_1"]
    modified = samples["damped_1"]
    cross_delta = modified[:, 3] - original[:, 3]
    summary = {
        "scene": str(scene),
        "mujoco": mujoco.__version__,
        "numpy": np.__version__,
        "joint": "joint1",
        "timestep_s": float(baseline.opt.timestep),
        "warmup_s": 0.0,
        "trajectory_duration_s": 0.4,
        "observation_duration_s": 0.8,
        "target_start_rad": float(original[0, 2]),
        "target_end_rad": float(original[-1, 2]),
        "baseline_damping": original_damping,
        "changed_damping": changed_damping,
        "damping_factor": args.damping_factor,
        "force_sampling": "mj_step force evaluation at control_time_s; state sampled at sample_time_s",
        "repeat_max_abs_position_delta_rad": {
            "baseline": float(np.max(np.abs(samples["baseline_2"][:, 3] - original[:, 3]))),
            "damped": float(np.max(np.abs(samples["damped_2"][:, 3] - modified[:, 3]))),
        },
        "condition_max_abs_position_delta_rad": float(np.max(np.abs(cross_delta))),
        "baseline_tracking_rmse_rad": float(np.sqrt(np.mean((original[:, 3] - original[:, 2]) ** 2))),
        "damped_tracking_rmse_rad": float(np.sqrt(np.mean((modified[:, 3] - modified[:, 2]) ** 2))),
        "baseline_final_position_rad": float(original[-1, 3]),
        "damped_final_position_rad": float(modified[-1, 3]),
        "samples": [
            {
                "time_s": float(original[index, 0]),
                "applied_target_rad": float(original[index, 2]),
                "baseline_position_rad": float(original[index, 3]),
                "damped_position_rad": float(modified[index, 3]),
                "condition_delta_rad": float(cross_delta[index]),
            }
            for index in (49, 99, 199, 399)
        ],
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "trajectory.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["condition", "sample_time_s", "control_time_s", "applied_target_rad",
                         "position_rad", "velocity_rad_s", "actuator_force", "joint_actuator_torque_Nm"])
        for name, values in samples.items():
            for row in values:
                writer.writerow([name, *row])
    report = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    (output / "summary.json").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
