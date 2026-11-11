"""Trace one fixed normalized action through the Panda environment and replay buffer."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

from panda_two_joint_env import PandaTwoJointEnv

SAC_DIR = Path(__file__).resolve().parents[1] / '11_SAC' / 'Pendulum'
sys.path.insert(0, str(SAC_DIR))
from rl_utils import ReplayBuffer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('--max-steps', type=int, default=250)
    args = parser.parse_args()

    env = PandaTwoJointEnv(args.scene, max_steps=args.max_steps)
    replay = ReplayBuffer(1)
    obs, _ = env.reset()
    # A fixed teaching input, not an output from a trained policy.
    action = np.array([1.0, -0.5])
    increment = 0.02 * action
    next_obs, reward, terminated, truncated, info = env.step(increment)
    # Match panda_sac_smoke.py: the last field masks only true termination.
    replay.add(obs, action, reward, next_obs, terminated)
    states, actions, rewards, next_states, dones = replay.sample(1)
    report = {
        'observation_order': ['q1', 'q2', 'dq1', 'dq2', 'goal1', 'goal2', 'ctrl1', 'ctrl2'],
        'state': states[0].tolist(),
        'normalized_action': actions[0].tolist(),
        'requested_increment_rad': increment.tolist(),
        'applied_increment_rad': info['applied_increment_rad'].tolist(),
        'reward': float(rewards[0]),
        'next_state': next_states[0].tolist(),
        'terminated': bool(terminated),
        'truncated': bool(truncated),
        'outcome': info['outcome'],
        'replay_done': bool(dones[0]),
        'reset_required': bool(terminated or truncated),
        'bootstrap_mask': int(1 - dones[0]),
    }
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
