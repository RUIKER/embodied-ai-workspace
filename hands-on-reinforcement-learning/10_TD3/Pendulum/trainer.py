import numpy as np
from tqdm import tqdm


def train_off_policy_agent(env, agent, num_episodes, replay_buffer,
                           minimal_size, batch_size, seed=None):
    """运行 TD3 的 off-policy 训练流程。

    循环结构：
        episode -> 与环境交互 -> 存入 replay buffer
                -> buffer 足够大后，每个环境步做一次 TD3 update

    TD3 的 update() 内部已经处理了 delayed policy update：
    每个环境步都会更新 critic，但每 policy_delay 步才更新一次 actor。
    """
    return_list = []
    with tqdm(total=num_episodes, desc='Training') as progress:
        for episode in range(num_episodes):
            # 每个 episode 使用不同 seed：实验整体可复现，
            # 但 Pendulum 的初始角度/速度不会每个 episode 完全相同。
            if seed is None:
                state, _ = env.reset()
            else:
                state, _ = env.reset(seed=seed + episode)

            done = False
            episode_return = 0.0

            while not done:
                # 训练时动作 = actor(s) + 高斯噪声，再裁剪到动作边界。
                action = agent.take_action(state, explore=True)
                next_state, reward, terminated, truncated, _ = env.step(action)

                # terminated 和 truncated 都会结束交互循环，
                # 但只有 terminated 才是“真正终止”。
                done = terminated or truncated
                # 时间上限造成的 truncated 不应停止 Bellman bootstrap。
                # 例如 Pendulum 每 200 步被截断，但摆仍然可以继续运动，
                # 因此写入 buffer 的 done 必须是 terminated。
                replay_buffer.add(
                    state, action, reward, next_state, terminated
                )
                state = next_state
                episode_return += reward

                # buffer 积累到 minimal_size 后开始学习。
                # 每个环境步更新一次，保持 1:1 的样本/梯度比例。
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


def evaluate_policy(env, agent, num_episodes=10, seed=None):
    """用不带探索噪声的确定性策略评估。

    训练回报包含探索噪声，不能代表策略本身的好坏。
    训练结束后关闭噪声再跑若干 episode，才是真正的 evaluation return。

    seed 不为 None 时，每个评估 episode 使用固定的初始状态。
    这样比较不同训练种子时，评估面对的是同一批初始状态，
    减少“运气好/坏”带来的评估方差。
    """
    returns = []
    for episode in range(num_episodes):
        if seed is None:
            state, _ = env.reset()
        else:
            state, _ = env.reset(seed=seed + episode)
        done = False
        episode_return = 0.0
        while not done:
            action = agent.take_action(state, explore=False)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_return += reward
        returns.append(episode_return)
    return np.asarray(returns, dtype=np.float32)
