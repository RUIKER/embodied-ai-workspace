import numpy as np
import torch


class GaussianNoise:
    """独立高斯探索噪声。

    TD3 的 actor 是确定性策略。为了让智能体在训练时尝试不同动作，
    行为动作使用：
        a_behavior = clip(mu(s) + sigma * epsilon, action_low, action_high)
    其中 epsilon ~ N(0, I)。

    与 DDPG 教程里的 OU 噪声不同，TD3 论文和 Spinning Up 默认使用
    独立高斯噪声：每一步重新采样，不保留时间相关性。TD3 的平滑性
    主要来自 target policy smoothing，而不是探索噪声。
    """

    def __init__(self, action_dim, sigma):
        self.action_dim = action_dim
        self.sigma = sigma

    def reset(self):
        # 独立高斯噪声没有历史状态；保留 reset 是为了和训练循环的接口统一。
        pass

    def sample(self):
        # 返回 float32，避免和 torch 的 float32 张量相加时发生类型提升。
        return self.sigma * np.random.randn(self.action_dim).astype(np.float32)


def smooth_target_action(actions, target_noise, noise_clip,
                         action_scale, action_low, action_high):
    """Target policy smoothing：给 target actor 的下一动作加截断噪声。

    公式（Spinning Up）：
        a'(s') = clip(mu_target(s') + clip(epsilon, -c, c), a_low, a_high)
        epsilon ~ N(0, sigma^2)

    为什么需要它？
    确定性 actor 的 target 动作只取决于 s'。如果 critic 在某个
    (s', mu_target(s')) 附近没有训练数据，它可能给出一个错误的尖峰 Q 值，
    而 actor 会利用这个“漏洞”把策略推向该动作。加入噪声后，critic 被要求
    在目标动作附近的一小片区域内都给出相近的 Q 值，从而平滑 Q 函数，
    减少策略利用 critic 误差的机会。

    注意单位：
    target_noise 和 noise_clip 按归一化动作空间 [-1, 1] 定义，
    这里乘以 action_scale 转成环境动作单位，再用真实动作边界裁剪。
    """
    # 与 actions 形状相同，每个样本、每个动作维度独立采样。
    noise = torch.randn_like(actions) * target_noise * action_scale
    # 截断噪声幅度，防止目标动作偏离 target actor 太远。
    noise = torch.clamp(
        noise, -noise_clip * action_scale, noise_clip * action_scale
    )
    # 最后再裁剪到环境合法动作范围，避免 critic 看到越界动作。
    return torch.max(torch.min(actions + noise, action_high), action_low)
