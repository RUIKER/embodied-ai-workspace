"""Compute the Panda hand pose from fixed body transforms and joint angles."""

import argparse
from pathlib import Path

import mujoco
import numpy as np


def quaternion_matrix(quaternion):
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def axis_rotation(axis, angle):
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def forward_hand(model, joint_angles):
    base = model.body("link0").id
    hand = model.body("hand").id
    joints = {model.joint(f"joint{i}").id: angle for i, angle in enumerate(joint_angles, 1)}
    chain = []
    body = hand
    while body != base:
        if body == 0:
            raise ValueError("The hand is not a descendant of link0")
        chain.append(body)
        body = model.body_parentid[body]

    transform = np.eye(4)
    seen_joints = set()
    for body in reversed(chain):
        local = np.eye(4)
        local[:3, :3] = quaternion_matrix(model.body_quat[body])
        local[:3, 3] = model.body_pos[body]
        transform = transform @ local
        start, count = model.body_jntadr[body], model.body_jntnum[body]
        for joint in range(start, start + count):
            if joint not in joints or model.jnt_type[joint] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError("Unexpected joint in the link0-to-hand chain")
            seen_joints.add(joint)
            pivot = model.jnt_pos[joint]
            angle = joints[joint] - model.qpos0[model.jnt_qposadr[joint]]
            rotation = axis_rotation(model.jnt_axis[joint], angle)
            motion = np.eye(4)
            motion[:3, :3] = rotation
            motion[:3, 3] = pivot - rotation @ pivot
            transform = transform @ motion
    if seen_joints != set(joints):
        raise ValueError("The link0-to-hand chain does not contain all seven arm joints")
    return transform


def check_pose(model, angles):
    joint_ids = [model.joint(f"joint{i}").id for i in range(1, 8)]
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    data.qpos[model.jnt_qposadr[joint_ids]] = angles
    mujoco.mj_forward(model, data)
    base, hand = model.body("link0").id, model.body("hand").id
    base_rotation = data.xmat[base].reshape(3, 3)
    actual_position = base_rotation.T @ (data.xpos[hand] - data.xpos[base])
    actual_rotation = base_rotation.T @ data.xmat[hand].reshape(3, 3)
    estimated = forward_hand(model, angles)
    position_error = np.linalg.norm(estimated[:3, 3] - actual_position)
    rotation_error = np.max(np.abs(estimated[:3, :3] - actual_rotation))
    return estimated, actual_position, actual_rotation, position_error, rotation_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path, help="Path to the pinned Panda scene.xml")
    parser.add_argument("--joints", type=float, nargs=7, metavar="RAD",
                        help="Seven joint positions in radians")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve(strict=True)))
    joint_ids = [model.joint(f"joint{i}").id for i in range(1, 8)]
    home = model.key("home").qpos[model.jnt_qposadr[joint_ids]].copy()
    if args.joints is not None:
        angles = np.asarray(args.joints)
        if not np.isfinite(angles).all():
            raise ValueError("Joint angles must be finite")
        cases = [("user", angles)]
    else:
        perturbed = home.copy()
        perturbed[0] += 0.1
        perturbed[1] -= 0.1
        cases = [("home", home), ("perturbed", perturbed)]
        for axis in range(7):
            single = home.copy()
            single[axis] += 0.05
            cases.append((f"joint{axis + 1}", single))

    for label, angles in cases:
        transform, actual_position, actual_rotation, position_error, rotation_error = check_pose(model, angles)
        print(f"case: {label}")
        print("joint_angles_rad:", np.array2string(angles, precision=8))
        print("FK hand position in link0 (m):", np.array2string(transform[:3, 3], precision=9))
        print("MuJoCo hand position in link0 (m):", np.array2string(actual_position, precision=9))
        print("FK hand rotation in link0:")
        print(np.array2string(transform[:3, :3], precision=9))
        print("MuJoCo hand rotation in link0:")
        print(np.array2string(actual_rotation, precision=9))
        print(f"position_error_m: {position_error:.3e}")
        print(f"rotation_max_element_error: {rotation_error:.3e}")
        if position_error > 1e-9 or rotation_error > 1e-9:
            raise RuntimeError(f"FK mismatch for {label}")


if __name__ == "__main__":
    main()
