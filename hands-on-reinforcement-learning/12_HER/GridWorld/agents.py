"""DDPG agent。HER 只改 replay buffer，不改算法本身。

======================================================================
DDPG 在本文件里要记住的四件事
======================================================================
1. 四个网络：actor μ(s)、critic Q(s,a)，各自还有一个 target 副本 μ'、Q'。
   target 副本不参与梯度，只负责算 Bellman 目标，用来抑制"自举"带来的发散。
   初始时 target 与 online 参数完全相同，之后靠软更新慢慢跟随。

2. critic 回归的目标（TD target）：
       y = r + γ · (1 − done) · Q'(s', μ'(s'))
   done 在这里只表示 terminated（到达目标），不包含 truncated（超时）。
   超时不是任务的终点，只是采样被截断，value 仍然应该往后自举。

3. actor 沿着"Q 对动作的梯度"上升（确定性策略梯度定理）：
       max_θ E[ Q(s, μ_θ(s)) ]     ⇒   loss_actor = −mean Q(s, μ_θ(s))
   梯度只回传到 actor，不回传到 critic（critic 的参数在 actor loss 里不更新，
   但因为两层计算图共用，实际实现上 actor 更新时 critic 也会算一次前向，
   只是不在这一步调用 critic 的 optimizer）。

4. target 软更新：θ' ← (1 − τ)·θ' + τ·θ，每个 batch 都动一点。
   τ = 0.005 意味着 target 大约 200 次更新跟随一次 online，变化足够平滑。

HER 不碰上面任何一条：它只改变采样时喂进来的 (s, a, r, s', done) 怎么造出来，
见 rl_utils.ReplayBufferTrajectory.sample。

======================================================================
与 09_DDPG/Pendulum/agents.py 结构相同，另外修了 practice.py 的两个问题
======================================================================
1. take_action 末尾强制 clip 到动作边界。
   practice.py 只加噪声不裁剪，buffer 里会存到越界动作（实测最大 1.313），
   而环境真正执行的是被 clip 后的值，critic 学到的是不一致的 (s, a)。
2. target 网络的计算放进 torch.no_grad()。
   practice.py 的 q_targets 带梯度，target 网络会累积用不到的 .grad，
   既浪费显存，也容易让人误以为 target 网络在被训练。
"""

import numpy as np
import torch
import torch.nn.functional as F

from networks import Actor, Critic


class DDPG:
    """确定性 actor + Q critic + 两个 target 网络。

    一次 update() 只做一次梯度下降；训练频率由 trainer.py 控制
    （每条轨迹结束后做 n_train 次），而不是每个 env step 做一次。
    """

    def __init__(self, state_dim, hidden_dim, action_dim, action_bound,
                 actor_lr, critic_lr, sigma, tau, gamma, device):
        """Args:
            state_dim:    观测维度 = 4（位置 2 + 目标 2）。
            hidden_dim:   隐藏层宽度。
            action_dim:   动作维度 = 2。
            action_bound: 动作上限，用于 tanh 输出缩放和 clip。
            actor_lr / critic_lr: 两个优化器的学习率，这里都是 1e-3。
            sigma:        探索噪声标准差（加在归一化动作上）。
            tau:          软更新系数。
            gamma:        折扣因子。
            device:       torch 设备。
        """
        # ① online 网络：参与梯度更新。
        self.actor = Actor(
            state_dim, hidden_dim, action_dim, action_bound
        ).to(device)
        self.critic = Critic(state_dim, hidden_dim, action_dim).to(device)

        # ② target 网络：结构完全相同，独立的一份参数。
        self.target_actor = Actor(
            state_dim, hidden_dim, action_dim, action_bound
        ).to(device)
        self.target_critic = Critic(
            state_dim, hidden_dim, action_dim
        ).to(device)

        # ③ target 初始时复制 online，之后只做软更新。
        #    如果不复制，target 一开始就是随机初始化，Bellman 目标会非常嘈杂。
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # ④ 两个优化器各自只管自己的网络。
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr
        )

        self.action_dim = action_dim
        self.action_bound = float(action_bound)
        self.sigma = float(sigma)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.device = device

    def take_action(self, state, explore=True):
        """根据状态选动作。

        Args:
            state: 形状 (4,) 的数组 [x, y, gx, gy]。
            explore: True  时加高斯噪声（训练采样用）；
                     False 时输出确定性动作（评估用）。
        Returns:
            形状 (2,) 的 float32 动作，已裁剪到 [−action_bound, action_bound]。

        注意两点：
        1. 噪声加在 tanh 之后的**归一化动作**上，所以 sigma=0.1 表示
           每维平均偏移 0.1，相对动作上限 1.0 是 10% 的扰动。
        2. 这里用的是全局 np.random（main.set_seed 里播种），
           而环境有自己独立的 Generator。两者是两条互不干扰的随机流。
        """
        # 网络的输入必须是 batch 形式，单条状态先 unsqueeze 成 (1, 4)，
        # 输出再 squeeze 回 (2,)。no_grad 是因为这里只做推理。
        state_t = torch.as_tensor(
            np.asarray(state), dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_t).squeeze(0).cpu().numpy()

        if explore:
            action = action + self.sigma * np.random.randn(self.action_dim)

        # 关键：buffer 里存的必须是环境实际执行的 action。
        # 环境的 step 里也会再 clip 一次，但那是"兜底"；如果不在这里裁，
        # buffer 里存的是未裁剪的动作、环境执行的是裁剪后的动作，
        # critic 就会学到一对在现实中永远不会出现的 (s, a)。
        action = np.clip(action, -self.action_bound, self.action_bound)
        return action.astype(np.float32)

    def soft_update(self, online_net, target_net):
        """target ← (1 − tau) · target + tau · online。

        用 mul_/add_ 原地修改，不建新张量；no_grad 是因为 target 的更新
        不是"优化"出来的，不该在计算图里留下痕迹。
        """
        with torch.no_grad():
            for target_param, online_param in zip(
                target_net.parameters(), online_net.parameters()
            ):
                target_param.mul_(1.0 - self.tau)
                target_param.add_(self.tau * online_param)

    def update(self, batch):
        """一个 batch 的 DDPG 更新；batch 是 rl_utils 返回的 dict。

        batch 的形状（B = batch_size，默认 256）：
            states       (B, 4)
            actions      (B, 2)
            rewards      (B,)      → 下面会 reshape 成 (B, 1)
            next_states  (B, 4)
            dones        (B,)      → 只含 terminated，不含 truncated

        更新顺序不能乱：
            算 target（无梯度）→ 更新 critic → 用更新后的 critic 更新 actor
            → 软更新两个 target
        """
        states = torch.as_tensor(
            batch['states'], dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            batch['actions'], dtype=torch.float32, device=self.device
        )
        # rewards / dones 是 (B,)，和 (B, 1) 的 Q 值做广播会得到 (B, B)，
        # 所以这里必须显式 view 成列向量。
        rewards = torch.as_tensor(
            batch['rewards'], dtype=torch.float32, device=self.device
        ).view(-1, 1)
        next_states = torch.as_tensor(
            batch['next_states'], dtype=torch.float32, device=self.device
        )
        dones = torch.as_tensor(
            batch['dones'], dtype=torch.float32, device=self.device
        ).view(-1, 1)

        # ---------------------------------------------------------- critic
        # Bellman target 不需要梯度：它只是一个回归目标（常数），
        # 若带着梯度反传，target 网络会收到本不该有的梯度。
        with torch.no_grad():
            next_actions = self.target_actor(next_states)            # (B, 2)
            next_q_values = self.target_critic(next_states, next_actions)  # (B, 1)
            # (1 − done)：终止状态不再有未来回报。
            # 因为 batch 里的 done 只来自 terminated，超时的样本会继续自举——
            # 这正是"截断"应有的处理方式。
            q_targets = rewards + self.gamma * next_q_values * (1.0 - dones)

        # 把 Q(s, a) 往 y 上回归。用 MSE 而不是 Huber，和教程保持一致；
        # 若奖励尺度很大或噪声多，可以换 smooth_l1_loss。
        critic_loss = F.mse_loss(self.critic(states, actions), q_targets)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ----------------------------------------------------------- actor
        # 确定性策略梯度：最大化 Q(s, μ(s))，所以 loss 取负号。
        # 这一步只调用 actor_optimizer，critic 的参数不会被动到。
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------------------------------------------------------- target
        # 两个 target 都软更新。放在最后，保证本 batch 用的是"上一版"的 target。
        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)
