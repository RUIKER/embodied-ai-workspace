import numpy as np
from tqdm import tqdm

from environment import continuous_to_discrete


def train_off_policy_agent(env, agent, num_episodes, replay_buffer,
                           minimal_size, batch_size, seed=None):
    """运行带连续动作代理器的CartPole off-policy训练。"""
    return_list = []
    with tqdm(total=num_episodes, desc='Training') as progress:
        for episode in range(num_episodes):
            agent.reset_noise()
            state, _ = env.reset() if seed is None else env.reset(seed=seed + episode)
            done = False
            episode_return = 0.0
            while not done:
                continuous_action = agent.take_action(state, explore=True)
                env_action = continuous_to_discrete(continuous_action)
                next_state, reward, terminated, truncated, _ = env.step(env_action)
                done = terminated or truncated
                # replay buffer 保存actor看到的连续动作，而非映射后的整数动作。
                # 时间上限截断不是真终止，target 仍然可以使用下一状态价值。
                replay_buffer.add(
                    state, continuous_action, reward, next_state, terminated
                )
                state = next_state
                episode_return += reward
                if replay_buffer.size() >= minimal_size:
                    agent.update(replay_buffer.sample(batch_size))
            return_list.append(episode_return)
            if (episode + 1) % 10 == 0:
                progress.set_postfix({
                    'episode': episode + 1,
                    'return': '%.3f' % np.mean(return_list[-10:]),
                })
            progress.update(1)
    return return_list
