"""Check the Menagerie Panda model without opening a rendering context."""

import argparse
import json
import platform
from pathlib import Path

import mujoco as mj
import numpy as np


def inspect_model(scene: Path) -> dict:
    model = mj.MjModel.from_xml_path(str(scene))
    data = mj.MjData(model)
    home = model.key("home")

    arm = []
    for number in range(1, 8):
        joint = model.joint(f"joint{number}")
        actuator = model.actuator(f"actuator{number}")
        if (model.actuator_trntype[actuator.id] != mj.mjtTrn.mjTRN_JOINT
                or model.actuator_trnid[actuator.id, 0] != joint.id):
            raise ValueError(f"{actuator.name} does not drive {joint.name}")
        arm.append({
            "joint": joint.name,
            "qpos_address": int(model.jnt_qposadr[joint.id]),
            "qvel_address": int(model.jnt_dofadr[joint.id]),
            "actuator": actuator.name,
            "ctrl_index": actuator.id,
            "ctrlrange_rad": model.actuator_ctrlrange[actuator.id].tolist(),
        })
    gripper = model.actuator("actuator8")
    if (model.actuator_trntype[gripper.id] != mj.mjtTrn.mjTRN_TENDON
            or model.actuator_trnid[gripper.id, 0] != model.tendon("split").id):
        raise ValueError("Expected actuator8 to drive the split tendon")

    arm_addresses = [axis["qpos_address"] for axis in arm]
    joint1_address = arm_addresses[0]
    joint1_ctrl = arm[0]["ctrl_index"]
    target = float(home.ctrl[joint1_ctrl] + 0.1)
    low, high = arm[0]["ctrlrange_rad"]
    if not low <= target <= high:
        raise ValueError("Joint 1 step target exceeds its actuator limits")

    runs = []
    for _ in range(2):
        mj.mj_resetDataKeyframe(model, data, home.id)
        mj.mj_forward(model, data)
        np.testing.assert_array_equal(data.qpos, home.qpos)
        np.testing.assert_array_equal(data.qvel, home.qvel)
        np.testing.assert_array_equal(data.ctrl, home.ctrl)
        np.testing.assert_equal(data.time, home.time[0])

        hold_drift = 0.0
        trajectory = []
        for step in range(250):
            if step == 100:
                data.ctrl[joint1_ctrl] = target
            mj.mj_step(model, data)
            state = np.concatenate((data.qpos, data.qvel, data.qacc))
            if not np.isfinite(state).all():
                raise RuntimeError(f"Non-finite state at step {step + 1}")
            if step < 100:
                drift = np.max(np.abs(data.qpos[arm_addresses] - home.qpos[arm_addresses]))
                hold_drift = max(hold_drift, float(drift))
            trajectory.append(state)
        np.testing.assert_allclose(data.time - home.time[0], 250 * model.opt.timestep,
                                   rtol=0, atol=1e-12)
        if any(warning.number for warning in data.warning):
            raise RuntimeError("MuJoCo reported a simulation warning")
        runs.append(np.asarray(trajectory))

    repeat_error = float(np.max(np.abs(runs[0] - runs[1])))
    np.testing.assert_allclose(runs[0], runs[1], rtol=0, atol=1e-12)
    joint1_end = float(data.qpos[joint1_address])
    if abs(joint1_end - target) >= abs(home.qpos[joint1_address] - target):
        raise RuntimeError("Joint 1 did not move closer to the commanded target")

    return {
        "python": platform.python_version(),
        "mujoco": mj.__version__,
        "numpy": np.__version__,
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "timestep_s": model.opt.timestep,
        "sim_time_s": float(data.time),
        "arm": arm,
        "gripper_ctrlrange": model.actuator_ctrlrange[gripper.id].tolist(),
        "home_qpos": home.qpos.tolist(),
        "home_qvel": home.qvel.tolist(),
        "home_ctrl": home.ctrl.tolist(),
        "hold_max_arm_drift_rad": hold_drift,
        "joint1_target_rad": target,
        "joint1_final_rad": joint1_end,
        "joint1_final_error_rad": abs(joint1_end - target),
        "repeat_max_abs_delta": repeat_error,
        "warnings": 0,
        "result": "PASS",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="Path to the pinned Panda scene.xml")
    args = parser.parse_args()
    print(json.dumps(inspect_model(args.scene), indent=2))
