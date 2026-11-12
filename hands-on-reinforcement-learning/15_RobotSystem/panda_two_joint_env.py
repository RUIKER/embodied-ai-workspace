"""Small, deterministic MuJoCo Panda two-joint reaching environment.

Native Python interface, not a Gymnasium environment or training implementation.
Actions are joint-target increments in radians; one step lasts 20 ms.
"""
import argparse
import json
from pathlib import Path
import mujoco
import numpy as np


class PandaTwoJointEnv:
    def __init__(self, scene, goal=(0.1, 0.1), max_steps=250):
        self.model = mujoco.MjModel.from_xml_path(str(Path(scene)))
        self.data = mujoco.MjData(self.model)
        self.joints = np.array([self.model.joint(f'joint{i}').id for i in (1, 2)])
        self.actuators = np.array([self.model.actuator(f'actuator{i}').id for i in (1, 2)])
        for joint, actuator in zip(self.joints, self.actuators):
            if (self.model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT
                    or self.model.actuator_trnid[actuator, 0] != joint):
                raise ValueError('Expected joint position actuators')
        if not np.isclose(self.model.opt.timestep, .002, atol=1e-12, rtol=0):
            raise ValueError('Expected 2 ms physics timestep')
        self.qadr = self.model.jnt_qposadr[self.joints]
        self.vadr = self.model.jnt_dofadr[self.joints]
        self.lower = np.maximum(self.model.jnt_range[self.joints, 0], self.model.actuator_ctrlrange[self.actuators, 0])
        self.upper = np.minimum(self.model.jnt_range[self.joints, 1], self.model.actuator_ctrlrange[self.actuators, 1])
        self.goal = self.validate_goal(goal)
        if not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError('max_steps must be a positive integer')
        self.max_steps = max_steps
        self.actual_lower = self.model.jnt_range[self.joints, 0] - 1e-3
        self.actual_upper = self.model.jnt_range[self.joints, 1] + 1e-3
        self.steps = 0
        self.finished = True

    def observation(self):
        return np.r_[self.data.qpos[self.qadr], self.data.qvel[self.vadr],
                     self.goal, self.data.ctrl[self.actuators]].copy()

    def validate_goal(self, goal):
        goal = np.array(goal, dtype=float, copy=True)
        if goal.shape != (2,) or not np.isfinite(goal).all() or np.any(goal < self.lower) or np.any(goal > self.upper):
            raise ValueError('Goal must contain two finite joint angles inside control limits')
        return goal

    def reset(self, *, goal=None):
        if goal is not None:
            self.goal = self.validate_goal(goal)
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key('home').id)
        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        self.finished = False
        return self.observation(), {'outcome': 'running'}

    def step(self, action):
        if self.finished:
            raise RuntimeError('Call reset before step or after an episode ends')
        action = np.asarray(action, dtype=float)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError('action must contain two finite increments')
        previous_target = self.data.ctrl[self.actuators]
        target = np.clip(previous_target + np.clip(action, -.02, .02), self.lower, self.upper)
        self.data.ctrl[self.actuators] = target
        for _ in range(10):
            mujoco.mj_step(self.model, self.data)
            invalid = not (np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all())
            warning = any(w.number for w in self.data.warning)
            q = self.data.qpos[self.qadr]
            # Allow small numerical excursions at MuJoCo's soft limits.
            out_of_range = np.any(q < self.actual_lower) or np.any(q > self.actual_upper)
            if invalid or warning or out_of_range:
                break
        self.steps += 1
        vel = self.data.qvel[self.vadr]
        failed = bool(invalid or warning or out_of_range)
        success = bool(not failed and np.all(np.abs(q - self.goal) < .01) and np.all(np.abs(vel) < .05))
        terminated = failed or success
        truncated = bool(not terminated and self.steps >= self.max_steps)
        self.finished = bool(terminated or truncated)
        outcome = 'failure' if failed else 'success' if success else 'timeout' if truncated else 'running'
        return self.observation(), (0.0 if success else -1.0), terminated, truncated, {
            'outcome': outcome, 'step': self.steps, 'applied_increment_rad': target - previous_target,
            'target_rad': target.copy(), 'physics_warning': bool(warning), 'actual_limit_violation': bool(out_of_range)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('--goal', type=float, nargs=2, default=(0.1, 0.1))
    parser.add_argument('--action', type=float, nargs=2, action='append', required=True)
    args = parser.parse_args()
    env = PandaTwoJointEnv(args.scene, goal=args.goal)
    observation, _ = env.reset()
    print(json.dumps({'event': 'reset', 'observation': observation.tolist()}))
    for action in args.action:
        observation, reward, terminated, truncated, info = env.step(action)
        print(json.dumps({'step': info['step'], 'observation': observation.tolist(),
                          'reward': reward, 'outcome': info['outcome'],
                          'terminated': terminated, 'truncated': truncated}))
        if terminated or truncated:
            break


if __name__ == '__main__':
    main()
