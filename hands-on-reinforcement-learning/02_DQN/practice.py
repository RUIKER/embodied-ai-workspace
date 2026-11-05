import random
import gymnasium as gym
import numpy as np
import collections
from tqdm import tqdm
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import rl_utils

class ReplayBuffer:
    '''经验回放池'''
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity) #队列，先进先出

    def add(self, state, action, reward, next_state, done): #将数据加入buffer
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size): #从buffer中采样数据, 数量为batch_size
        transitions = random.sample(self.buffer, batch_size) #随机采样
        states, actions, rewards, next_states, dones = zip(*transitions) #解压
        return np.array(states), actions, rewards, np.array(next_states), dones

    def size(self): #目前buffer中数据的数量
        return len(self.buffer)

class Qnet(torch.nn.Module):
    '''只有一层隐藏层的Q网络'''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim) #输入层到隐藏层
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim) #隐藏层到输出层

    def forward(self, x): #前向传播
        x = F.relu(self.fc1(x)) #隐藏层使用ReLU激活函数
        return self.fc2(x)

class DQN:
    '''DQN算法'''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma, epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim, action_dim).to(device) #Q网络
        self.target_q_net = Qnet(state_dim, hidden_dim, action_dim).to(device) #目标Q网络
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=learning_rate) #Adam优化器
        self.gamma = gamma #折扣因子
        self.epsilon = epsilon #epsilon-greedy
        self.target_update = target_update #目标网络更新频率
        self.count = 0 #计数器，记录更新次数
        self.device = device #设备

    def take_action(self, state): #epsilon-greedy采取策略
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim) #随机选择动作
        else:
            state = torch.tensor([state], dtype = torch.float).to(self.device) #将状态转换为tensor
            action = self.q_net(state).argmax().item() #选择Q值最大的动作
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['rewards'], dtype = torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype = torch.long).view(-1, 1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype = torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype = torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype = torch.float).view(-1, 1).to(self.device)
        q_values = self.q_net(states).gather(1, actions) #当前状态的Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(-1, 1) #下一个状态的最大Q值
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones) #TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets)) #均方误差损失函数
        self.optimizer.zero_grad() #Pytorch中默认梯度会累积，这里需要显式将梯度设置为0
        dqn_loss.backward() #反向传播
        self.optimizer.step() #更新参数

        if self.count % self.target_update == 0: #每隔target_update步更新目标网络
            self.target_q_net.load_state_dict(self.q_net.state_dict()) #将Q网络的参数复制到目标Q网络

        self.count += 1