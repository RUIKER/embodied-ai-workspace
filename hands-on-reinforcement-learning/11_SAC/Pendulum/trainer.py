import numpy as np
from tqdm import tqdm


def train_off_policy_agent(env, agent, num_episodes, replay_buffer,
                           minimal_size, batch_size, seed=None):
    """执行 SAC 的 off-policy 训练。

    和 PPO/TRPO 不同，SAC 不要求数据来自当前策略；
    transition 会长期保存在 replay buffer 中并被重复采样。
    """
    return_list = []
    with tqdm(total=num_episodes, desc='Training') as progress:
        for episode in range(num_episodes):
            # 每个 episode 可以使用不同 seed，保证实验可复现但初始状态不完全相同。
            state, _ = env.reset() if seed is None else env.reset(seed=seed + episode)
            done = False
            episode_return = 0.0
            while not done:
                # SAC actor 自带随机性，不需要额外的 epsilon/OU 噪声。
                # SAC actor 本身是随机策略，不需要额外的 epsilon 或 OU 噪声。
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                # done 控制交互循环；terminated 才写入 buffer 作为真正 terminal。
                # truncated 通常只是时间上限，Bellman target 仍然可以 bootstrap。
                replay_buffer.add(state, action, reward, next_state, terminated)
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
