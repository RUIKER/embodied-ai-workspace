import numpy as np
from tqdm import tqdm


def train_on_policy_agent(env, agent, num_episodes, episodes_per_update=1,
                          seed=None):
    """采集 episodes_per_update 个完整回合后更新一次 TRPO。

    transition_dict 里 `dones` 只记录真正的终止（terminated），用于 bootstrap；
    `ends` 记录回合边界（terminated 或 truncated），用于切断 GAE。
    Pendulum 永远不会 terminated，200 步截断处仍要用 V(s') 继续 bootstrap。
    """
    return_list = []
    with tqdm(total=num_episodes, desc='Training') as progress:
        episode = 0
        while episode < num_episodes:
            transition_dict = {
                'states': [], 'actions': [], 'next_states': [],
                'rewards': [], 'dones': [], 'ends': [],
            }
            for _ in range(min(episodes_per_update, num_episodes - episode)):
                if seed is None:
                    state, _ = env.reset()
                else:
                    state, _ = env.reset(seed=seed + episode)
                done = False
                episode_return = 0.0
                while not done:
                    # 连续策略返回形如 [action_dim] 的浮点动作向量。
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    # TRPO 保存实际采样出的 action，后续用它计算 old_log_probs。
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(terminated)
                    transition_dict['ends'].append(done)
                    state = next_state
                    episode_return += reward
                return_list.append(episode_return)
                episode += 1
                progress.update(1)
            agent.update(transition_dict)
            progress.set_postfix({
                'episode': episode,
                'return': '%.3f' % np.mean(return_list[-10:]),
            })
    return return_list
