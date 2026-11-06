# MuJoCo 与 Panda 执行器

> 轨迹点给出关节位置目标，MuJoCo 按位置执行器、动力学和积分步长推进状态。固定输入的无渲染冒烟已在 Windows 与 WSL 通过。

## 1. 三个量分别是什么

| 量 | 在 Panda 中怎么读 |
| --- | --- |
| `data.qpos` | 关节位置。臂轴转角用 rad，从 `model.jnt_qposadr[joint.id]` 取地址。 |
| `data.qvel` | 广义速度。臂轴角速度用 rad/s，从 `model.jnt_dofadr[joint.id]` 取地址。 |
| `data.ctrl` | 执行器输入，按 actuator ID 索引。臂轴接收位置目标，夹爪接收驱动 `split` 腱的模型专用命令。 |

解析 `joint1` 与 `actuator1` 的名字后，核对 `actuator_trntype` 和 `actuator_trnid`，再保存关节地址与执行器索引。这份模型的臂轴地址恰好相同，换模型后仍应分别查表。两根手指由同一个腱执行器驱动。

## 2. 写入目标后发生什么

位置执行器根据目标与反馈生成作用力，惯性、重力、阻尼和约束共同决定下一步状态。写 `ctrl` 后调用 `mj_step`，读取新 `qpos` 和 `qvel`。`ctrlrange` 约束控制输入，关节行程和执行器力限制应分别核对。

从 `home` 关键帧重置后调用 `mj_forward` 更新派生量，按同一输入顺序重复推进。记录模型 SHA、依赖、物理步长、初态和目标。脚本和固定输入协议见 [Panda 冒烟](../../hands-on-reinforcement-learning/15_RobotSystem/README.md)。

## 3. 自测

- **改参数前先预测**。在冒烟脚本的临时副本中，将 `joint1` 的目标偏移量反号，确认目标仍在 `ctrlrange` 内。预测状态响应的方向，再比较目标与末值。末帧误差只描述结束时刻。
- **故障定位**。收到角度目标却朝错误关节运动，依次检查消息关节顺序、关节名到运行时地址的映射、执行器传动关系和角度单位。
- **面试一句话**。我用固定版本 Panda 跑通了 `home` 重置和单轴阶跃，检查了关节映射、状态有限性和同进程重复性，也在 WSL 验证了 ROS2 到 Panda 的固定轨迹链路。见 [工程验收](../11_工程主线验收.md)。

参考 [MuJoCo 仿真文档](https://mujoco.readthedocs.io/en/stable/programming/simulation.html)。
