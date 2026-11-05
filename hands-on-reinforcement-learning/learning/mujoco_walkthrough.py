"""MuJoCo / HalfCheetah 上手脚本：先读输出，再按 TODO 自己改。

运行：.venv/Scripts/python.exe learning/mujoco_walkthrough.py
对应周六学习块；目的是把 XML、MuJoCo 底层 API、Gymnasium 封装三层对上。
"""

import mujoco
import numpy as np
import gymnasium as gym

# ---------- 1. 环境封装层（Gymnasium） ----------
env = gym.make('HalfCheetah-v5')
u = env.unwrapped
obs, _ = env.reset(seed=0)
print('== Gymnasium 层 ==')
print('obs', obs.shape, 'action', env.action_space.shape,
      'range', env.action_space.low[0], env.action_space.high[0])
print('frame_skip', u.frame_skip, '| timestep', u.model.opt.timestep,
      '| 控制步长 dt =', u.dt)
print('最长回合步数', env.spec.max_episode_steps)

# ---------- 2. 模型层（MjModel：XML 编译后的静态结构） ----------
m, d = u.model, u.data
print('\n== MjModel 层 ==')
print('nq(位置自由度)', m.nq, '| nv(速度自由度)', m.nv, '| nu(执行器)', m.nu)
print('关节：', [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(m.njnt)])
print('执行器 gear：', m.actuator_gear[:, 0].tolist())
print('执行器 ctrlrange：', m.actuator_ctrlrange[0].tolist())
# TODO 1：obs 的 17 维 = qpos[1:] (8) + qvel (9)。解释为什么 qpos 去掉了第 0 维（rootx）？

# ---------- 3. 数据层（MjData：每一步变化的状态） ----------
print('\n== MjData 层 ==')
print('qpos', np.round(d.qpos, 3))
print('qvel', np.round(d.qvel, 3))
x0 = d.qpos[0]
total_ctrl = 0.0
for _ in range(100):
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    total_ctrl += info['reward_ctrl']
print('随机动作 100 步：前进 %.3f m, 累计 ctrl reward %.3f' % (d.qpos[0] - x0, total_ctrl))
print('info keys:', sorted(info.keys()))
# TODO 2：用 reward_forward 与 reward_ctrl 验证 reward = forward - 0.1 * ||a||^2。
# TODO 3：直接调用 mujoco.mj_step(m, d)（不经过 env.step）连续 5 次，对比 env.step 一次，
#         确认 frame_skip=5 的含义。
# TODO 4：把 d.ctrl 设成全 0，看 20 步内 qpos 怎么变；再设成全 1。
env.close()
