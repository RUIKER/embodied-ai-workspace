"""Solve Panda hand-origin position in link0 with damped least squares."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from panda_fk import check_pose, forward_hand


class HandPositionIK:
    def __init__(self, model):
        self.model = model
        self.data = mujoco.MjData(model)
        mujoco.mj_resetDataKeyframe(model, self.data, model.key("home").id)
        joints = [model.joint(f"joint{i}").id for i in range(1, 8)]
        actuators = [model.actuator(f"actuator{i}").id for i in range(1, 8)]
        self.qadr = model.jnt_qposadr[joints]
        self.dadr = model.jnt_dofadr[joints]
        self.home = self.data.qpos[self.qadr].copy()
        self.lower = np.maximum(model.jnt_range[joints, 0], model.actuator_ctrlrange[actuators, 0])
        self.upper = np.minimum(model.jnt_range[joints, 1], model.actuator_ctrlrange[actuators, 1])
        self.hand = model.body("hand").id
        self.base = model.body("link0").id
        self.jacobian = np.empty((3, model.nv))

    def position(self, q):
        self.data.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, self.data)
        base_rotation = self.data.xmat[self.base].reshape(3, 3)
        position = base_rotation.T @ (self.data.xpos[self.hand] - self.data.xpos[self.base])
        return position, base_rotation

    def solve(self, goal, initial=None):
        goal = np.asarray(goal, dtype=float)
        q = self.home.copy() if initial is None else np.array(initial, dtype=float, copy=True)
        if goal.shape != (3,) or not np.isfinite(goal).all():
            raise ValueError("Expected a finite three-dimensional target in link0")
        if q.shape != (7,) or not np.isfinite(q).all() or np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError("Initial angles must lie within all seven joint and actuator limits")

        tolerance, damping, max_iterations = 1e-4, 0.02, 150
        reason, clipped_steps = "iteration_limit", 0
        for iteration in range(max_iterations + 1):
            position, base_rotation = self.position(q)
            error = goal - position
            error_norm = np.linalg.norm(error)
            mujoco.mj_jacBody(self.model, self.data, self.jacobian, None, self.hand)
            jac = base_rotation.T @ self.jacobian[:, self.dadr]
            if error_norm <= tolerance:
                reason = "converged"
                break
            if iteration == max_iterations:
                break

            delta = jac.T @ np.linalg.solve(jac @ jac.T + damping**2 * np.eye(3), error)
            delta *= min(1.0, 0.15 / max(np.max(np.abs(delta)), 1e-12))
            for scale in (1, 0.5, 0.25, 0.125, 0.0625, 0.03125):
                proposal = q + scale * delta
                trial = np.clip(proposal, self.lower, self.upper)
                trial_position, _ = self.position(trial)
                if np.linalg.norm(goal - trial_position) < error_norm - 1e-12:
                    clipped_steps += int(np.any(trial != proposal))
                    q = trial
                    break
            else:
                reason = "no_descent"
                break

        fk, _, _, position_difference, _ = check_pose(self.model, q)
        fk_error = float(np.linalg.norm(fk[:3, 3] - goal))
        if position_difference > 1e-9:
            raise RuntimeError("Independent forward kinematics disagrees with the IK model")
        return {
            "goal_link0_m": goal.tolist(), "joint_angles_rad": q.tolist(),
            "fk_position_link0_m": fk[:3, 3].tolist(), "fk_error_m": fk_error,
            "status": reason, "iterations": iteration, "clipped_steps": clipped_steps,
            "at_joint_limit": (np.isclose(q, self.lower, atol=1e-7, rtol=0)
                               | np.isclose(q, self.upper, atol=1e-7, rtol=0)).tolist(),
            "min_position_jacobian_singular_value": float(np.linalg.svd(jac, compute_uv=False)[-1]),
        }


def smoke(solver):
    offsets = (
        ("home", np.zeros(7)),
        ("reachable_a", np.array([0.20, -0.18, 0.10, 0.15, 0., 0.1, 0.])),
        ("reachable_b", np.array([-0.22, 0.16, -0.12, -0.2, 0.08, 0., 0.])),
    )
    cases = {
        label: solver.solve(forward_hand(solver.model, solver.home + offset)[:3, 3])
        for label, offset in offsets
    }
    cases["distant_goal"] = solver.solve([3., 0., 3.])
    cases["joint_limit"] = solver.solve([0., 0., -3.])
    for label, result in cases.items():
        if (result["status"] == "converged") != (label in dict(offsets)):
            raise RuntimeError(f"Unexpected IK smoke outcome for {label}: {result}")
        angles = np.asarray(result["joint_angles_rad"])
        if not np.isfinite(angles).all() or np.any(angles < solver.lower) or np.any(angles > solver.upper):
            raise RuntimeError(f"Invalid joint angles for {label}")
    limit = cases["joint_limit"]
    if not limit["clipped_steps"] or not any(limit["at_joint_limit"]):
        raise RuntimeError("Expected the joint-limit smoke case to exercise clipping")
    return {"status": "PASS", "mujoco": mujoco.__version__, "numpy": np.__version__, "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--goal", nargs=3, type=float, metavar="M")
    modes.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, help="New JSON output file")
    args = parser.parse_args()
    model = mujoco.MjModel.from_xml_path(str(args.scene.resolve(strict=True)))
    solver = HandPositionIK(model)
    result = smoke(solver) if args.smoke else solver.solve(args.goal)
    report = json.dumps(result, indent=2, allow_nan=False)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(report + "\n")
    print(report)
    if args.goal is not None and result["status"] != "converged":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
