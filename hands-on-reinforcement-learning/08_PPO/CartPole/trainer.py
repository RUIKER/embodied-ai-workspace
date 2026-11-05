import numpy as np
from tqdm import tqdm


def train_on_policy_agent(env, agent, num_episodes):
    """训练 PPO。

    PPO 是 on-policy 算法：当前策略采集一个完整 episode，
    使用这段数据更新若干个 epoch，然后丢弃这段旧数据。
    """
    return_list = []
    with tqdm(total=num_episodes, desc='Training') as progress:
        for episode in range(num_episodes):
            # Gymnasium 的 reset 返回 (observation, info)。
            state, _ = env.reset()
            done = False
            episode_return = 0.0
            transition_dict = {
                'states': [], 'actions': [], 'next_states': [],
                'rewards': [], 'dones': [],
            }
            while not done:
                # actor 根据当前策略分布采样动作，而不是使用 argmax。
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                # terminated 表示环境自然结束，truncated 通常表示达到时间上限。
                done = terminated or truncated

                # 一个 transition 是 (s, a, r, s', done)。
                # PPO 需要整条轨迹来计算 GAE，而不是立即更新。
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward
            agent.update(transition_dict)
            return_list.append(episode_return)
            if (episode + 1) % 10 == 0:
                progress.set_postfix({
                    'episode': episode + 1,
                    'return': '%.3f' % np.mean(return_list[-10:]),
                })
            progress.update(1)
    return return_list
