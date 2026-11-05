import numpy as np
from tqdm import tqdm


def train_off_policy_agent(env, agent, num_episodes, replay_buffer,
                           minimal_size, batch_size, seed=None):
    """运行DDPG的off-policy训练流程。"""
    return_list = []
    with tqdm(total=num_episodes, desc='Training') as progress:
        for episode in range(num_episodes):
            # OU噪声需要在每个episode开始时清零历史状态。
            agent.reset_noise()
            if seed is None:
                state, _ = env.reset()
            else:
                state, _ = env.reset(seed=seed + episode)
            done = False
            episode_return = 0.0

            while not done:
                # actor给出确定性动作，noise负责探索。
                action = agent.take_action(state, explore=True)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                # truncated 通常只是达到时间上限，不代表真实终止。
                # Bellman target 在时间截断处仍应 bootstrap 下一状态价值。
                terminal_for_target = terminated
                replay_buffer.add(
                    state, action, reward, next_state, terminal_for_target
                )
                state = next_state
                episode_return += reward

                # buffer积累足够数据后，每个环境步都可采样更新。
                if replay_buffer.size() >= minimal_size:
                    batch = replay_buffer.sample(batch_size)
                    agent.update(batch)

            return_list.append(episode_return)
            if (episode + 1) % 10 == 0:
                progress.set_postfix({
                    'episode': episode + 1,
                    'return': '%.3f' % np.mean(return_list[-10:]),
                })
            progress.update(1)
    return return_list
