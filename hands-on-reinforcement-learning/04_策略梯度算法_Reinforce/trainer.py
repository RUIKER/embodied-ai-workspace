import numpy as np
from tqdm import tqdm

def train_on_policy_agent(env, agent, num_episodes):
    '''运行REINFORCE训练：每个episode结束后更新一次策略网络。'''
    return_list = []
    with tqdm(total=num_episodes, desc='Training') as pbar:
        for i_episode in range(num_episodes):
            episode_return = 0
            transition_dict = {
                'states': [],
                'actions': [],
                'next_states': [],
                'rewards': [],
                'dones': [],
            }
            state, _ = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                transition_dict['states'].append(state)
                transition_dict['actions'].append(action)
                transition_dict['next_states'].append(next_state)
                transition_dict['rewards'].append(reward)
                transition_dict['dones'].append(done)
                state = next_state
                episode_return += reward

            # REINFORCE使用完整episode的回报更新一次策略。
            agent.update(transition_dict)
            return_list.append(episode_return)
            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode': i_episode + 1,
                    'return': '%.3f' % np.mean(return_list[-10:]),
                })
            pbar.update(1)
    return return_list