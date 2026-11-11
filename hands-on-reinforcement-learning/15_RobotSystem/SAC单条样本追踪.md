# Panda SAC 单条样本追踪

这个练习追踪一次 `状态 → 动作 → 环境反馈 → 回放池`。先预测再运行，最后用自己的话解释；不需要训练模型。下面的固定动作是教学输入，不是训练好的策略输出。本人运行与口述尚待完成。

## 先预测

从 home 重置后，任务目标是 `(0.1, 0.1)` rad，两轴执行器目标为 `(0, 0)` rad。假设归一化动作是 `(1, -0.5)`，对照 [SAC 短检查采样循环](panda_sac_smoke.py)：

1. 乘以 `0.02` 后，传给环境的动作是多少？这两个数是关节角、任务目标，还是目标增量？
2. 如果没有碰到限位，新的执行器目标是多少？实际关节角是否必然立即等于它？
3. 记录一条 transition 时，回放池里的动作是乘之前还是乘之后的值？

## 运行并逐字段核对

在 Ubuntu WSL 终端运行，使用项目已有的 MuJoCo 环境和固定模型：

```bash
project='/mnt/c/Users/PC/Desktop/Embodied_Inteligence_Internship/hands-on-reinforcement-learning/15_RobotSystem'
scene='/mnt/c/Users/PC/AppData/Local/EmbodiedAI/PandaSmoke/menagerie-98a70c5/franka_emika_panda/scene.xml'
panda_python='/var/tmp/embodied-panda-ros2-lyrical/ros-venv/bin/python'

"$panda_python" "$project/panda_transition_trace.py" "$scene"
```

输出中的 `state` 和 `next_state` 均按 `observation_order` 排列：两轴实际角度 q（rad）、实际角速度 dq（rad/s）、任务目标 goal（rad）、执行器目标 ctrl（rad）。`normalized_action` 是回放池保存的动作；`requested_increment_rad` 是环境收到的增量，`applied_increment_rad` 是环境经过增量和命令限幅后实际施加的目标变化。输出的 `reward`、`replay_done` 和两条状态也从真实回放池的单条抽样中取得。这里的样本只有一条，因此抽样不会挑到其他样本。

顺着代码找对应位置：

- [采样循环](panda_sac_smoke.py)：`take_action(obs)` 得到归一化动作；`env.step(0.02 * action)` 执行一步；`replay.add(obs, action, reward, next_obs, terminated)` 存五个字段。当前短检查的 `take_action` 是随机采样；本练习用固定输入代替它，便于手算。
- [环境](panda_two_joint_env.py)：`observation()` 组装八个字段；`step()` 先限制增量和目标，再推进物理仿真，按实际位置和速度判断结果。正常完成一步推进 10 个 2 ms 物理步，若发现异常则可能提前中断。
- [回放池](../11_SAC/Pendulum/rl_utils.py)：按 `(state, action, reward, next_state, done)` 保存；这里 `done` 由调用方传入 `terminated`。`truncated` 和 `outcome` 没有单独存入该回放池。
- [SAC 更新](../11_SAC/Pendulum/agents.py)：抽出的 `dones` 用于目标中的 `(1 - dones)`，因此决定是否保留下一状态价值项。此脚本只采样、不执行更新。

<details>
<summary>Agent 本机参考输出与解释，先自行预测再展开</summary>

2026-11-10，agent 使用 README 中固定模型、Windows CPU 环境、MuJoCo 3.17.0 与 NumPy 2.5.4 执行，动作固定且无随机采样。初始 `state` 为 `(0, 0, 0, 0, 0.1, 0.1, 0, 0)`；归一化动作 `(1, -0.5)` 转为增量 `(0.02, -0.01)` rad。限幅没有改变该增量，步后 `ctrl` 为 `(0.02, -0.01)` rad。下一状态中实际 q 约为 `(0.003600, -0.000598)` rad，dq 约为 `(0.166862, -0.029388)` rad/s，goal 仍为 `(0.1, 0.1)` rad。`reward=-1`、`terminated=false`、`truncated=false`，回放池 `done=false`。

这里执行器目标是控制命令，实际关节位置由仿真动力学推进后得到，并非立即等于目标。奖励为 -1 是因为这一步尚未满足两轴的位置和速度成功条件。

</details>

## 比较超时与终止

再将本次练习的回合上限改为一步，其余输入保持不变：

```bash
"$panda_python" "$project/panda_transition_trace.py" "$scene" --max-steps 1
```

观察 `terminated`、`truncated`、`reset_required`、`replay_done` 和 `bootstrap_mask`，回答：为什么此时要 reset，目标计算却仍保留下一状态价值项？在当前实现中，成功或失败使用 `terminated`，外部步数上限使用 `truncated`；只用前者屏蔽 bootstrap。下一状态是超时前最后一步的状态，随后才 reset。这个选择依赖把步数上限视作外部截断；若任务本身定义了有限时域，终止语义和状态设计需另行判断。

<details>
<summary>Agent 对照结果</summary>

该固定输入在第一步未成功也未失败。`--max-steps 1` 时 `terminated=false`、`truncated=true`、`outcome=timeout`；`reset_required=true`，但 `replay_done=false`、`bootstrap_mask=1`。正常上限的第一步两个结束标记均为 false，其他状态、动作和奖励相同。

</details>

## 口述检查

请用一两分钟讲清：`state` 的八个值分别是什么；一次固定动作怎样变成执行器目标；环境根据什么给奖励；回放池存什么；正常、成功和超时分别怎样处理 reset 和 bootstrap。可把实际输出及你的解释发给 agent 核对；不要把 agent 的验证记作本人已复现。
