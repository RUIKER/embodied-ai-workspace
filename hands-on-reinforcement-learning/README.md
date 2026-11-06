# Reinforcement Learning 实验代码

这是一个强化学习学习项目，按“表格型方法 → 深度价值方法 → 策略梯度 → 连续控制”的路线组织。代码主要使用 Python、Gymnasium、NumPy、PyTorch 和 Matplotlib。

目录编号表示**推荐阅读顺序**，不是严格的历史时间顺序。算法之间的继承与分支关系见 [算法路线.md](算法路线.md)。

## 项目目录

| 编号 | 目录 | 内容 | 入口 |
| ---: | --- | --- | --- |
| 01 | `01_Cliff Walking` | Cliff Walking 上的表格型 Q-learning 和 SARSA | `train.py` |
| 02 | `02_DQN` | DQN 最小练习实现（Q 网络、replay buffer、target network） | `practice.py`（参考代码） |
| 03 | `03_Double DQN、Dueling DQN` | DQN、Double DQN、Dueling DQN 统一 pipeline | `main.py` |
| 04 | `04_策略梯度算法_Reinforce` | CartPole-v1 上的 REINFORCE | `main.py` |
| 05 | `05_Mountain Car` | 基于 RBF 特征的线性 Actor-Critic | `main.py` |
| 06 | `06_Mountain Car PyTorch` | 基于 PyTorch MLP 的 Actor-Critic | `main.py` |
| 07 | `07_TRPO` | CartPole 离散动作与 Pendulum 连续动作 TRPO | `*/main.py` |
| 08 | `08_PPO` | CartPole 离散动作与 Pendulum 连续动作 PPO | `*/main.py` |
| 09 | `09_DDPG` | Pendulum 标准 DDPG 与 CartPole 教学适配 | `*/main.py` |
| 10 | `10_TD3` | Pendulum 连续动作 TD3 | `Pendulum/main.py` |
| 11 | `11_SAC` | Pendulum 连续动作 SAC | `Pendulum/main.py` |
| 12 | `12_HER` | 二维目标导向任务上的 DDPG + HER（replay buffer 改造） | `GridWorld/main.py` |
| 13 | `13_Imitation` | 模仿学习：BC 最小实现（GridWorld / 双走廊），BC / ACT / Diffusion Policy 算法卡片 | `bc_minimal.py` |
| 14 | `14_Control/Pendulum` | 有模型的经典控制对照线：LQR、能量摆起、参数扰动与执行延迟 | `lqr.py`、`evaluate.py` |
| 15 | [`15_RobotSystem`](15_RobotSystem/README.md) | Panda 仿真与 ROS2 轨迹、状态和录包回放冒烟 | `panda_smoke.py`、`ros2_smoke/run.py` |

每个算法目录下的 `经验总结.md` 都采用同一结构：先回答“特点、针对的缺陷、局限性、应用场景”四个问题，再记录实现细节、实验结果和排错经验。

> `12_HER` 不是新的算法分支，而是 DDPG 在目标导向任务上的 replay buffer 改造；它与 DDPG 的逐段差异和 10-seed 实测结果见 [`12_HER/经验总结.md`](12_HER/经验总结.md)。

## 算法推进线（速览）

```text
价值型分支
01 表格型 Q-learning / SARSA
        │  状态空间爆炸
        ▼
02 DQN ──► 03 Double DQN / Dueling DQN
        │  离散动作、间接学策略
        ▼
策略梯度分支
04 REINFORCE ──► 05 Actor-Critic ──► 06 Actor-Critic (PyTorch)
                                      │  步长敏感、更新可能崩溃
                                      ▼
                              07 TRPO ──► 08 PPO

连续控制 off-policy 分支
02 DQN 的 replay + target network
08 PPO 的连续动作策略经验
        │
        ▼
09 DDPG ──► 10 TD3 ──► 11 SAC
```

更完整的对比表、每一条箭头的含义，以及“为什么编号不是一条直线”，见 [算法路线.md](算法路线.md)。

## 对比实验（`experiments/`）

`experiments/` 在统一协议下比较 09–11（DDPG / TD3 / SAC）与 07–08 的 Pendulum 版本（TRPO / PPO）：
**10 个固定 seed、20 个固定评估起始状态、确定性评估**；off-policy 跑 40k 步，on-policy 跑 400k 步。

![五算法学习曲线：中位数 ± IQR，10 个 seed](experiments/seed_experiments/five_algorithms/results/fig1_learning_curves.png)

| 算法 | 最终评估中位数 | 首次达到 -200 的步数 |
| --- | ---: | ---: |
| SAC | -121.2 | 5k |
| TD3 | -123.1 | 10k |
| DDPG | -127.9 | 8k |
| PPO | -129.7 | 95k |
| TRPO | -132.5 | 90k |

五个算法最终水平相当，差别在样本效率和稳定性；这组实验也定位并修复了第一版 PPO/TRPO 的实现问题（更新批太小、
时间截断当终止、TRPO 的 KL 约束过小）。用法和图表见 [experiments/README.md](experiments/README.md)，
结论见 [experiments/seed_experiments/five_algorithms/经验总结.md](experiments/seed_experiments/five_algorithms/经验总结.md)。

## 经典控制对照（`14_Control/Pendulum`）

在同一批固定评估起点上，加了一条"有模型"的对照组：能量摆起 + LQR。
名义条件 20/20、平均回报 -128.5；被控对象质量或长度偏移 ±30% 时仍是 20/20，
但 3 步（0.15 s）执行延迟下崩到 0/20；LQR 单独只有 7/20。
口径、调参方式和边界见 [14_Control/Pendulum/README.md](14_Control/Pendulum/README.md)。

## 代码结构可视化

仓库提供一份单文件的交互式代码流页面：[强化学习代码流](code-flow.html)。页面回答一个问题：这些目录和文件是怎么连起来的。

- 一条算法目录的调用链：`main.py → config.py → environment.py → agents.py → trainer.py → visualization.py`；
- 11 个目录各自的入口与训练循环；
- off-policy 与 on-policy 的更新时机差异（每步 / 每回合 / 每 10 回合）；
- `experiments/` 用子进程横切 5 个算法的完整流水线；
- 2026-10 这次改写的 PPO / TRPO v2 对照表。

文件引用带行号，指向提交 `fbe0e5f`；页面不依赖外部脚本或网络，直接用浏览器打开即可。

## 环境准备

推荐使用 Python 3.10 或更高版本，并为项目创建虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
```

根目录的 `requirements.txt` 覆盖全部实验，一次装好即可：

```bash
python -m pip install -r requirements.txt
```

每个实验目录也保留了自己的 `requirements.txt`（01 只用标准库，05/06 额外需要 scikit-learn），只想跑单个算法时可以只装对应目录的。

## 运行示例

### 01 Cliff Walking

只使用 Python 标准库：

```bash
cd "01_Cliff Walking"
python train.py
```

在 `train.py` 中切换 `q_learning.QTable` 和 `sarsa.SarsaTable` 可以对比 Q-learning 与 SARSA。

### 03 Double DQN、Dueling DQN

```bash
cd "03_Double DQN、Dueling DQN"
python -m pip install -r requirements.txt
python main.py
```

在 `config.py` 中修改 `ALGORITHM`，可选 `DQN`、`DoubleDQN`、`DuelingDQN`。

### 04 REINFORCE

```bash
cd "04_策略梯度算法_Reinforce"
python -m pip install -r requirements.txt
python main.py
```

### 05 / 06 Mountain Car

```bash
cd "05_Mountain Car"
python -m pip install -r requirements.txt
python main.py
```

```bash
cd "06_Mountain Car PyTorch"
python -m pip install -r requirements.txt
python main.py
```

### 07 TRPO / 08 PPO

```bash
cd "07_TRPO/CartPole"
python main.py

cd "08_PPO/Pendulum"
python main.py
```

Pendulum 版本已更新为 v2（每 10 个回合更新一次等），改动前的代码在各自的 `Pendulum/legacy_v1/`，原因见 `experiments/seed_experiments/five_algorithms/经验总结.md` §4。

### 09 DDPG / 10 TD3 / 11 SAC

```bash
cd "09_DDPG/Pendulum"
python main.py

cd "10_TD3/Pendulum"
python main.py

cd "11_SAC/Pendulum"
python main.py
```

## 参考与致谢

本项目是学习与复现项目，不是任何论文、教材或开源项目的官方实现。代码按“每个算法一个独立 pipeline”的方式重新组织，并适配了 Gymnasium 的新版接口；具体的拆分方式、实现细节和超参数起点参考了以下资料。

### 教程与参考实现

- **《动手学强化学习》**，张伟楠、沈键、俞勇，人民邮电出版社，2022：[在线教程](https://hrl.boyuai.com/) · [配套代码](https://github.com/boyu-ai/Hands-on-RL)。本仓库的 `config.py` / `environment.py` / `networks.py` / `agents.py` / `trainer.py` / `main.py` 拆分方式，以及 `rl_utils.py` 中的 replay buffer、`moving_average`、`compute_advantage` 等工具，参考了该教程的组织方式。
- **OpenAI Spinning Up**：[教程主页](https://spinningup.openai.com/) · [TD3 教程](https://spinningup.openai.com/en/latest/algorithms/td3.html) · [TD3 PyTorch 参考实现](https://github.com/openai/spinningup/tree/master/spinup/algos/pytorch/td3)。`10_TD3` 的算法结构、默认噪声参数和 `practice.py` 的对照写法参考了这里。
- **Gymnasium 文档**：[https://gymnasium.farama.org/](https://gymnasium.farama.org/)。环境接口、`reset()` / `step()` 的返回值，以及 `terminated` / `truncated` 的语义，以官方文档为准。

### 算法论文

| 编号 | 算法 | 原始论文 / 主要参考 |
| ---: | --- | --- |
| 01 | 表格型 Q-learning / SARSA | [Watkins & Dayan, *Q-learning*, Machine Learning, 1992](https://doi.org/10.1007/BF00992698)；[Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed., 2018](http://incompleteideas.net/book/the-book-2nd.html) |
| 02 | DQN | [Mnih et al., *Human-level control through deep reinforcement learning*, Nature, 2015](https://doi.org/10.1038/nature14236) |
| 03 | Double DQN / Dueling DQN | [van Hasselt et al., *Deep Reinforcement Learning with Double Q-Learning*, AAAI 2016](https://arxiv.org/abs/1509.06461)；[Wang et al., *Dueling Network Architectures for Deep Reinforcement Learning*, ICML 2016](https://arxiv.org/abs/1511.06581) |
| 04 | REINFORCE | [Williams, *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*, Machine Learning, 1992](https://doi.org/10.1007/BF00992696) |
| 05–06 | Actor-Critic | [Sutton & Barto, 2018](http://incompleteideas.net/book/the-book-2nd.html)，第 13 章；具体实现参考《动手学强化学习》 |
| 07 | TRPO | [Schulman et al., *Trust Region Policy Optimization*, ICML 2015](https://arxiv.org/abs/1502.05477)；优势估计使用 [GAE, Schulman et al., ICLR 2016](https://arxiv.org/abs/1506.02438) |
| 08 | PPO | [Schulman et al., *Proximal Policy Optimization Algorithms*, 2017](https://arxiv.org/abs/1707.06347) |
| 09 | DDPG | [Lillicrap et al., *Continuous Control with Deep Reinforcement Learning*, ICLR 2016](https://arxiv.org/abs/1509.02971) |
| 10 | TD3 | [Fujimoto et al., *Addressing Function Approximation Error in Actor-Critic Methods*, ICML 2018](https://arxiv.org/abs/1802.09477) |
| 11 | SAC | [Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*, ICML 2018](https://arxiv.org/abs/1801.01290)；自动温度系数参考 [*Soft Actor-Critic Algorithms and Applications*, 2018](https://arxiv.org/abs/1812.05905) |

### 工具与依赖

- [Python](https://www.python.org/)
- [Gymnasium](https://gymnasium.farama.org/)（Farama Foundation）
- [NumPy](https://numpy.org/)
- [PyTorch](https://pytorch.org/)
- [Matplotlib](https://matplotlib.org/)

需要第三方依赖的实验目录通过 `requirements.txt` 记录实际版本；本仓库不包含上述库的源代码。

### 许可证说明

- 《动手学强化学习》配套代码以 [Apache License 2.0](https://github.com/boyu-ai/Hands-on-RL/blob/main/LICENSE) 发布。
- OpenAI Spinning Up 以 [MIT License](https://github.com/openai/spinningup/blob/master/LICENSE) 发布。

本仓库中的相关实现是在这些公开代码基础上重新拆分、改写并适配 Gymnasium 的。再分发或公开发布时，请保留原项目的版权、许可证和 NOTICE 声明（如有），并注明本项目做了哪些修改。

### 致谢

感谢《动手学强化学习》的作者和社区提供公开课程与配套代码，感谢上海交通大学 APEX 实验室和伯禹学习平台对相关课程与代码的公开支持，感谢 OpenAI Spinning Up 团队对 TD3 等算法的清晰讲解和参考实现，也感谢 Gymnasium、PyTorch、NumPy 和 Matplotlib 等开源项目的维护者。没有这些公开资料和工具，本项目难以完成。

### 引用说明

如果本项目帮助你完成了课程作业、论文或开源项目，建议引用对应算法的原始论文；如果参考了本仓库的代码组织，也请一并引用《动手学强化学习》。本仓库中的实现经过重新拆分、改写和 Gymnasium 适配，若与原始论文或参考实现存在差异，应以原始论文和当前代码为准。

## 说明

- 每个实验的 `经验总结.md` 记录算法原理、实现细节、排错过程和后续改进方向。
- TD3 只适用于连续动作空间，因此只提供 `Pendulum-v1` 案例。
- `.venv`、`__pycache__`、`.DS_Store` 和 `.pt` 权重由根目录 `.gitignore` 排除；`experiments/seed_experiments/five_algorithms/results/` 里的 `raw/*.json`、图和汇总表会保留。
- Gymnasium 项目使用新版 `reset()` 和 `step()` 接口；旧版 Gym 示例需要适配。
