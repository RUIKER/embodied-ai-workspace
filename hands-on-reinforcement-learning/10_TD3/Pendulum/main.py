import random

import numpy as np
import torch

import config
from agents import TD3
from environment import make_environment
from rl_utils import ReplayBuffer
from trainer import evaluate_policy, train_off_policy_agent
from visualization import plot_training_results


def set_seed(seed):
    """固定 Python、NumPy、PyTorch 的随机种子，方便复现实验。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed(config.SEED)

    env, state_dim, action_dim, action_low, action_high = make_environment(
        config.ENV_NAME
    )
    env.reset(seed=config.SEED)

    agent = TD3(
        state_dim=state_dim,
        hidden_dim=config.HIDDEN_DIM,
        action_dim=action_dim,
        action_low=action_low,
        action_high=action_high,
        actor_lr=config.ACTOR_LR,
        critic_lr=config.CRITIC_LR,
        gamma=config.GAMMA,
        tau=config.TAU,
        policy_delay=config.POLICY_DELAY,
        target_noise=config.TARGET_NOISE,
        noise_clip=config.NOISE_CLIP,
        exploration_sigma=config.EXPLORATION_SIGMA,
        device=config.DEVICE,
    )

    replay_buffer = ReplayBuffer(config.BUFFER_SIZE)

    return_list = train_off_policy_agent(
        env=env,
        agent=agent,
        num_episodes=config.NUM_EPISODES,
        replay_buffer=replay_buffer,
        minimal_size=config.MINIMAL_SIZE,
        batch_size=config.BATCH_SIZE,
        seed=config.SEED,
    )

    # 训练曲线中的回报包含探索噪声；这里再记录一次无噪声评估。
    # 使用独立且固定的评估种子：所有训练种子面对同一批初始状态。
    eval_returns = evaluate_policy(
        env, agent,
        num_episodes=config.EVAL_EPISODES,
        seed=config.EVAL_SEED,
    )
    print('最后 10 回合训练平均回报: {:.3f}'.format(
        np.mean(return_list[-10:])
    ))
    print('确定性策略评估回报: 均值 {:.3f}, 标准差 {:.3f}, 最佳 {:.3f}'.format(
        eval_returns.mean(), eval_returns.std(), eval_returns.max()
    ))

    plot_training_results(return_list, config.ENV_NAME)
    env.close()


if __name__ == '__main__':
    main()
