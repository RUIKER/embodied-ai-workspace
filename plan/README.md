# 计划与复习入口

[《作战计划 v5》](00_作战计划.docx)决定本轮范围与优先级。每周先看部署清单，执行后把证据记入进度追踪，再用复习卡口述一个链路、预测一个参数变化或复盘一次故障。

## 计划与执行

| 用途 | 文件 |
| --- | --- |
| 看本周做什么 | [W2 部署清单](01_本周部署清单_W2.docx) |
| 查工程范围、验收条件和最新证据 | [工程主线验收](11_工程主线验收.md)、[Panda 冒烟与复现](../hands-on-reinforcement-learning/15_RobotSystem/README.md) |
| 记录完成状态 | [进度追踪](10_进度追踪.xlsx) |
| 开始一个工作块 | [工程提示词](05_ClaudeCode提示词.docx) |

## 复习卡片

| 建议顺序 | 卡片 | 复习时回答 |
| --- | --- | --- |
| W2 | [01 MuJoCo 与 Panda 执行器](12_复习卡片/01_MuJoCo与Panda.md) | 写入 `ctrl` 后，为什么 `qpos` 会逐步变化？ |
| W3 | [02 ROS2 构建与通信](12_复习卡片/02_ROS2构建与通信.md) | C++ 发布器能看到，Python 为什么收不到？ |
| W4 | [03 轨迹执行与时钟](12_复习卡片/03_轨迹执行与时钟.md) | 仿真暂停时，如何检测命令停发？ |
| W5 | [04 录包回放与评估](12_复习卡片/04_录包回放与评估.md) | 同一个 bag 能分别证明数据链路和执行复现吗？ |
| 面试基础 | [05 VLA 基础](12_复习卡片/05_VLA基础.md) | 语言如何影响动作，动作又怎样交给控制器？ |

已有素材按原目录复习。

- 模仿学习见 [BC](../hands-on-reinforcement-learning/13_Imitation/算法卡片_BC.md)、[ACT](../hands-on-reinforcement-learning/13_Imitation/算法卡片_ACT.md)、[Diffusion Policy](../hands-on-reinforcement-learning/13_Imitation/算法卡片_DP.md)。实践证据见 [BC 最小实现与结果](../hands-on-reinforcement-learning/13_Imitation/README.md)。
- 强化学习见 [算法路线](../hands-on-reinforcement-learning/算法路线.md)、[五算法评估协议](../hands-on-reinforcement-learning/experiments/seed_experiments/five_algorithms/README.md)、[HER 经验总结](../hands-on-reinforcement-learning/12_HER/经验总结.md) 和 [SAC 经验总结](../hands-on-reinforcement-learning/11_SAC/经验总结.md)。引用实验数值前先查对应评估协议和 seed。
- 经典控制见 [Pendulum 对照](../hands-on-reinforcement-learning/14_Control/Pendulum/README.md)，重点复习 LQR 有效域、能量摆起、参数扰动和执行延迟。
- Search Agent 的检索、重排、MCP 和评测口径见 [项目 README](../industrial-search-agent/README.md) 与 [评估协议](../industrial-search-agent/docs/eval-protocol.md)。当前可讲虚构语料冒烟及工程链路。

## 面试与投递

- 技术自测与过往答题记录见 [面试自测](06_面试拷打点.docx) 和 [每周反思与关键问答](09_每周反思与关键问答.docx)。
- 岗位筛选见 [公司岗位表](07_公司岗位表.csv) 和 [JD 对照表](08_JD对照表.csv)。投递前逐岗核验在招状态、到岗日期、每周出勤、实习月数和通勤。
