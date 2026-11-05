import numpy as np
from tqdm import tqdm

from environment import dis_to_con


def train_DQN(agent, env, num_episodes, replay_buffer, minimal_size, batch_size):
    '''执行统一训练流程，agent内部决定使用哪一种DQN变体。'''
    return_list = []
    max_q_value_list = []
    max_q_value = 0
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset()
                done = False
                while not done:
                    # 1. agent根据当前状态选择离散动作。
                    action = agent.take_action(state)
                    max_q_value = agent.max_q_value(state) * 0.005 + max_q_value * 0.995
                    max_q_value_list.append(max_q_value)

                    # Pendulum本身需要连续动作；这里把离散动作映射回连续区间。
                    action_continuous = dis_to_con(action, env, agent.action_dim)
                    next_state, reward, terminated, truncated, _ = env.step([action_continuous])
                    done = terminated or truncated

                    # 2. 保存经验(s, a, r, s', done)，暂时不立即更新网络。
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward

                    # 3. buffer积累足够样本后，随机采样batch交给agent.update()。
                    # 随机采样能打散相邻时刻的相关性，提高训练稳定性。
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states': b_s,
                            'actions': b_a,
                            'next_states': b_ns,
                            'rewards': b_r,
                            'dones': b_d
                        }
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                        'return': '%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    return return_list, max_q_value_list