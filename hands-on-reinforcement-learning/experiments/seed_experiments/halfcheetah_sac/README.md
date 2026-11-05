# HalfCheetah SAC（v5）

MuJoCo HalfCheetah 上的 SAC，3 seeds × 1M steps。复用 `11_SAC/Pendulum/agents.py`
的 SAC 类，本地 `networks.py` 把 actor/critic 换成 2×256 MLP（MuJoCo 常用配置）。

## 协议

| 项 | 设置 |
| --- | --- |
| 环境 | `HalfCheetah-v5`（obs 17，action 6，动作范围 [-1,1]） |
| 算法 | SAC，hidden 256，batch 256，buffer 1e6，gamma .99，tau .005 |
| 学习率 | actor / critic / alpha = 3e-4 |
| 训练 | 1M env steps，每步更新一次；前 10k 步只填 buffer |
| 评估 | 每 20k 步 5 回合；最终 20 回合；确定性 `tanh(mu)` |
| seeds | 0 / 1 / 2（可扩到 5） |
| 设备 | CUDA（RTX 5060 Ti） |

## 运行

```bash
cd experiments/seed_experiments/halfcheetah_sac
bash run_all.sh
../../../.venv/Scripts/python.exe aggregate.py
```

输出：`results/raw/sac_s*.json`、`results/learning_curves.png`、
`results/final_eval.png`、`results/summary.csv`、`results/SUMMARY.md`。

## 超参覆盖（供 `ablations/sac_hparams` 使用）

`run_one.py` 支持 `--hidden-dim --actor-lr --critic-lr --alpha-lr --tau --gamma
--batch-size --reward-scale --init-alpha`，默认值来自 `settings.py`；实际使用值写入 JSON 的
`config`。基线 1M run 使用默认值，初始 alpha 为 0.01（`11_SAC/Pendulum/agents.py` 写死）。

## 环境说明

- `.venv` 已安装 `gymnasium[mujoco]`（mujoco 3.17）。
- 目录已经改成英文路径后，MuJoCo 可以直接读取 assets；不再需要 ASCII 资产补丁。
- `HalfCheetah-v4` 已 deprecated，统一用 v5。

## 结果（3 seeds × 1M steps，已完成）

| seed | final mean（20 回合） | final median | final best | 训练耗时 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 11083.6 | 11063.9 | 11256.9 | 14047s |
| 1 | 11437.9 | 11451.6 | 11572.6 | 14042s |
| 2 | 10230.8 | 10230.8 | 10562.2 | 14068s |

均值 10917.4，中位数 11083.6。曲线在 32–52 万步达到最终值的 80%，
之后缓慢上升/平台，1M 时没有明显崩塌。产物见 `results/SUMMARY.md`、
`results/summary.csv`、`results/learning_curves.png`、`results/final_eval.png`。

final mean 来自 20 个固定评估回合（`EVAL_SEED=10000`），动作取确定性的 `tanh(mu)`。
与公开数字比较时需要对齐环境版本、评估回合数和动作选择方式，见下一节。

## 与公开参考值对比

参考值来自 rl-baselines3-zoo 的 `benchmark.md`：1M 步训练，确定性策略，150 个评估回合，每个环境 1 次训练。

| 来源 | 环境 | 超参 | 最终回报 | 评估回合 |
| --- | --- | --- | ---: | ---: |
| 本仓库 SAC | HalfCheetah-v5 | lr 3e-4、τ 0.005、γ 0.99、batch 256、2×256、UTD 1 | 10917（3 seeds 均值） | 20 |
| SB3 rl-zoo SAC | HalfCheetah-v3 | SB3 默认（lr 3e-4、τ 0.005、γ 0.99、batch 256、2×256） | 9535 ± 100（1 次训练） | 150 |

本仓库比参考值高约 14%。

可比的部分：

- 超参一致。zoo 的 `hyperparams/sac.yml` 中 HalfCheetah-v4 只设置了 `learning_starts: 10000`，其余沿用 SB3 的 SAC 默认值。
- 环境版本。按 gymnasium 自带的 changelog，v3 到 v5 的变化在接口层（info 键名、`frame_skip` 和 `xml_file` 参数、`observation_structure`），没有改动力学和奖励，回报量级可比。

未排除的差异，因此该对比不能用来说明本实现更优：

- 训练次数与评估回合：参考值为 1 次训练、150 回合；本仓库为 3 个 seed、每个 20 回合。
- 底层绑定：v3 使用 mujoco-py，v5 使用新绑定，数值差异未验证。
- 初始 alpha：SB3 从 1.0 起步，本实现从 0.01 起步，影响见 `experiments/ablations/sac_hparams`。

归因需要在同一环境和评估协议下重跑 SB3，本仓库没有做。

## 设计说明

### 不使用 HER 的原因

HER 面向稀疏二值奖励的目标到达任务：多数 transition 的奖励相同（-1），通过把目标换成实际到达的状态来获得学习信号。
HalfCheetah 的奖励每一步都带信息（`1.0 × x_velocity − 0.1 × ‖a‖²`），不存在失败样本没有信号的问题，
也没有天然的目标 g 和映射 φ(s)。

如果强行套用，需要改写成目标条件任务，例如目标为期望速度 v*，奖励为 `−|x_velocity − v*|`，
用回合中实际达到的速度作为 hindsight 目标。得到的是跟踪给定速度的策略，与最大化前进速度不是同一个任务，本实验不做。

### 与 Pendulum 配置的差异

| 项 | Pendulum（`11_SAC/Pendulum`） | HalfCheetah（本目录） | 决定因素 |
| --- | --- | --- | --- |
| target entropy | -1 | -6 | 动作维度，取 -dim(A) |
| 网络 | 1 个隐藏层，128 | 2 个隐藏层，256 | 经验选择 |
| reward shift / scale | 8 / 8 | 0 / 1 | 奖励量级：Pendulum 单步奖励约在 [-16, 0]，需要平移缩放；HalfCheetah 随训练从约 0 增长到约 11，不需要 |
| critic lr | 3e-3 | 3e-4 | 调参结果；3e-4 是 MuJoCo SAC 的常用值 |
| batch / buffer | 64 / 1e5 | 256 / 1e6 | 训练预算，经验选择 |
| 训练步数 | 约 2 万（100 回合 × 200 步） | 100 万 | 任务难度 |
