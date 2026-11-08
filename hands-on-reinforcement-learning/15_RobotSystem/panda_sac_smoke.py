"""Verify the Panda → replay buffer → SAC → checkpoint path on CPU."""
import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch

from panda_two_joint_env import PandaTwoJointEnv

SAC_DIR = Path(__file__).resolve().parents[1] / '11_SAC' / 'Pendulum'
sys.path.insert(0, str(SAC_DIR))
from agents import SAC
from rl_utils import ReplayBuffer


NETWORKS = ('actor', 'critic_1', 'critic_2', 'target_critic_1', 'target_critic_2')
OPTIMIZERS = ('actor_optimizer', 'critic_1_optimizer', 'critic_2_optimizer', 'log_alpha_optimizer')
CONFIG = dict(state_dim=8, hidden_dim=64, action_low=[-1., -1.], action_high=[1., 1.],
              actor_lr=3e-4, critic_lr=3e-4, alpha_lr=3e-4, target_entropy=-2.,
              tau=.005, gamma=.99, reward_shift=0., reward_scale=1., device='cpu')


def checkpoint(agent):
    return {'config': CONFIG, 'networks': {n: getattr(agent, n).state_dict() for n in NETWORKS},
            'optimizers': {n: getattr(agent, n).state_dict() for n in OPTIMIZERS},
            'log_alpha': agent.log_alpha.detach().clone()}


def reload_agent(path):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    agent = SAC(**saved['config'])
    for n, state in saved['networks'].items():
        getattr(agent, n).load_state_dict(state)
    for n, state in saved['optimizers'].items():
        getattr(agent, n).load_state_dict(state)
    with torch.no_grad():
        agent.log_alpha.copy_(saved['log_alpha'])
    return agent


def run(scene, output):
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    agent = SAC(**CONFIG)
    env = PandaTwoJointEnv(scene, max_steps=16)
    replay = ReplayBuffer(64)
    obs, _ = env.reset()
    reference = obs.copy()
    outcomes = {'success': 0, 'failure': 0, 'timeout': 0}
    for _ in range(64):
        action = agent.take_action(obs)
        if action.shape != (2,) or not np.isfinite(action).all() or np.any(np.abs(action) > 1):
            raise RuntimeError('SAC action must be finite and normalized')
        next_obs, reward, terminated, truncated, info = env.step(0.02 * action)
        if not np.isfinite(next_obs).all() or not np.isfinite(reward):
            raise RuntimeError('Non-finite transition')
        # Timeouts reset the rollout but still bootstrap in the critic target.
        replay.add(obs, action, reward, next_obs, terminated)
        obs = next_obs
        if terminated or truncated:
            outcomes[info['outcome']] += 1
            obs, _ = env.reset()

    modules = {n: getattr(agent, n) for n in NETWORKS}
    before = {n: [p.detach().clone() for p in net.parameters()] for n, net in modules.items()}
    agent.update(replay.sample(32))
    delta = {n: max((p.detach() - b).abs().max().item()
                    for p, b in zip(net.parameters(), before[n])) for n, net in modules.items()}
    if any(not np.isfinite(d) or d <= 0 for d in delta.values()) or not torch.isfinite(agent.log_alpha):
        raise RuntimeError(f'Invalid SAC update: {delta}')

    output.mkdir(parents=True, exist_ok=False)
    path = output / 'sac_cpu.pt'
    torch.save(checkpoint(agent), path)
    restored = reload_agent(path)
    for name, net in modules.items():
        for key, value in net.state_dict().items():
            torch.testing.assert_close(value, getattr(restored, name).state_dict()[key], rtol=0, atol=0)
    for name in OPTIMIZERS:
        torch.testing.assert_close(getattr(agent, name).state_dict(),
                                   getattr(restored, name).state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(agent.log_alpha, restored.log_alpha, rtol=0, atol=0)
    torch.manual_seed(seed + 1)
    original_action = agent.take_action(reference)
    torch.manual_seed(seed + 1)
    restored_action = restored.take_action(reference)
    np.testing.assert_array_equal(restored_action, original_action)

    summary = {'status': 'PASS', 'device': 'cpu', 'torch': str(torch.__version__),
               'seed': seed, 'transitions': replay.size(), 'episode_outcomes': outcomes,
               'updates': 1, 'batch_size': 32, 'max_parameter_change': delta,
               'reloaded_action_max_abs_difference': float(np.max(np.abs(restored_action - original_action))),
               'checkpoint': str(path.resolve())}
    report = json.dumps(summary, indent=2, allow_nan=False)
    (output / 'summary.json').write_text(report + '\n', encoding='utf-8')
    print(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path)
    parser.add_argument('output', type=Path, help='New output directory')
    args = parser.parse_args()
    run(args.scene, args.output)


if __name__ == '__main__':
    main()
