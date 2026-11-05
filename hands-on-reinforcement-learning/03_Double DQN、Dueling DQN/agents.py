import numpy as np
import torch
import torch.nn.functional as F

from networks import DuelingQnet, Qnet


class DQN:
    '''统一的DQN agent，算法差异由dqn_type控制。'''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device, dqn_type='DQN'):
        self.action_dim = action_dim
        # DQN和DoubleDQN使用普通Q网络；DuelingDQN只替换网络结构。
        # 三者仍然共享下面的动作选择、经验回放和参数更新流程。
        network = DuelingQnet if dqn_type == 'DuelingDQN' else Qnet
        self.q_net = network(state_dim, hidden_dim, self.action_dim).to(device)
        self.target_q_net = network(state_dim, hidden_dim, self.action_dim).to(device)
        # target network只用于计算训练目标，不直接参与梯度更新。
        self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.count = 0
        self.device = device
        self.dqn_type = dqn_type

    def take_action(self, state):
        # epsilon-greedy：偶尔随机探索，避免只重复当前看起来最好的动作。
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            # q_net输出每个动作的Q值，argmax选择估计回报最高的动作。
            action = self.q_net(state).argmax().item()
        return action

    def max_q_value(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()

    def update(self, transition_dict):
        # replay buffer中的一条经验是(s, a, r, s', done)；这里一次取一个batch。
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device)

        # q_net(states)包含所有动作的Q值，gather只取实际执行动作a对应的Q(s,a)。
        current_q_values = self.q_net(states).gather(1, actions)

        # 计算下一状态的估计价值。它只是监督学习目标，所以不需要梯度。
        if self.dqn_type == 'DoubleDQN':
            with torch.no_grad():
                # Double DQN：在线网络负责“选动作”，目标网络负责“评估动作”。
                max_action = self.q_net(next_states).max(1)[1].view(-1, 1)
                next_q_values = self.target_q_net(next_states).gather(1, max_action)
        else:
            with torch.no_grad():
                # 普通DQN/DuelingDQN：目标网络同时选择并评估最大Q值动作。
                next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1)

        # Bellman目标：Q_target = r + gamma * Q(s', a')。
        # done为1时表示回合结束，不再使用下一状态价值。
        td_targets = rewards + self.gamma * next_q_values * (1 - dones)
        dqn_loss = F.mse_loss(current_q_values, td_targets)

        # 只更新online network；target network每隔target_update步整体同步一次。
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        if self.count % self.target_update == 0:
            # 延迟同步可以让训练目标在一段时间内保持稳定。
            self.target_q_net.load_state_dict(self.q_net.state_dict())
        self.count += 1


def build_agent(algorithm, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
    # main.py只调用这个工厂函数，因此切换算法不需要复制训练代码。
    # VanillaDQN是旧配置名，这里保留兼容性并统一转换为DQN。
    aliases = {'VanillaDQN': 'DQN'}
    dqn_type = aliases.get(algorithm, algorithm)
    if dqn_type not in {'DQN', 'DoubleDQN', 'DuelingDQN'}:
        raise ValueError(
            'Unsupported algorithm: {}. Choose DQN, DoubleDQN, or DuelingDQN.'.format(algorithm)
        )

    return DQN(state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device, dqn_type)