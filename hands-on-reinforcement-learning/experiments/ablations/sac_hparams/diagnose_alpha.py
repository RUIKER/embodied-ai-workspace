"""诊断：初始 alpha 0.01 vs 1.0 在前 30k 步里 alpha / 熵 / 策略均值如何变化。

sweep 里 `alpha1.0` 在 20k 步时全部低于基线，这里验证“高初始温度让策略先走向随机、
再被自动调温拉回”这一解释，而不是只靠推测。只跑 seed 0，30k 步，每 1k 步记录一次。
"""

import argparse
import json
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
HC = HERE.parents[1] / 'seed_experiments' / 'halfcheetah_sac'
SAC_DIR = HERE.parents[2] / '11_SAC' / 'Pendulum'
sys.path.insert(0, str(SAC_DIR))
sys.path.insert(0, str(HC))  # 本地 2×256 networks 必须排在前面

import settings  # noqa: E402
from agents import SAC  # noqa: E402
from rl_utils import ReplayBuffer  # noqa: E402


def probe(agent, buffer):
    """在一个 batch 上测：alpha、策略熵估计、|tanh(mu)|、Q 均值。"""
    states, _, _, _, _ = buffer.sample(512)
    s = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=agent.device)
    with torch.no_grad():
        actions, log_probs = agent.actor(s)
        hidden = F.relu(agent.actor.fc2(F.relu(agent.actor.fc1(s))))
        mu = torch.tanh(agent.actor.fc_mu(hidden))
        q = torch.min(agent.critic_1(s, actions), agent.critic_2(s, actions))
    return {
        'alpha': float(agent.log_alpha.exp()),
        'entropy': float(-log_probs.mean()),      # 目标是 -6（= target_entropy 的相反数）
        'mean_abs_mu': float(mu.abs().mean()),     # 评估用 tanh(mu)，接近 0 就是“几乎不动”
        'q_mean': float(q.mean()),
    }


def run(init_alpha, steps, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = gym.make(settings.ENV_ID)
    dim = env.action_space.shape[0]
    agent = SAC(
        env.observation_space.shape[0], settings.HIDDEN_DIM,
        env.action_space.low.copy(), env.action_space.high.copy(),
        settings.ACTOR_LR, settings.CRITIC_LR, settings.ALPHA_LR,
        -dim, settings.TAU, settings.GAMMA, 0.0, 1.0, settings.DEVICE,
    )
    agent.log_alpha.data.fill_(float(np.log(init_alpha)))
    buffer = ReplayBuffer(settings.BUFFER_SIZE)
    trace = []
    obs, _ = env.reset(seed=seed)
    for step in range(1, steps + 1):
        action = agent.take_action(obs)
        nxt, r, term, trunc, _ = env.step(action)
        buffer.add(obs, action, r, nxt, term)
        obs = nxt
        if term or trunc:
            obs, _ = env.reset()
        if buffer.size() >= settings.MINIMAL_SIZE:
            agent.update(buffer.sample(settings.BATCH_SIZE))
        if step % 1000 == 0 and buffer.size() >= settings.MINIMAL_SIZE:
            trace.append({'step': step, **probe(agent, buffer)})
    return trace


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--init-alpha', type=float, required=True)
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    out = HERE / 'results' / 'diagnostics'
    out.mkdir(parents=True, exist_ok=True)
    trace = run(args.init_alpha, args.steps, args.seed)
    (out / f'alpha_trace_a{args.init_alpha}_s{args.seed}.json').write_text(
        json.dumps(trace, indent=1), encoding='utf-8')
    print('done', args.init_alpha, len(trace))
