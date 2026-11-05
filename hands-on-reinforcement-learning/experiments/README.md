# experiments

实验按用途分成两组，算法代码仍在各自目录，实验脚本和结果集中在这里。

## seed_experiments — 之前的 seed 主实验

| 目录 | 内容 | 入口 |
| --- | --- | --- |
| `five_algorithms/` | DDPG / TD3 / SAC / PPO / TRPO 五算法 × 10 seeds，Pendulum-v1 | `five_algorithms/README.md`、`five_algorithms/经验总结.md` |
| `her_main/` | DDPG vs DDPG+HER，10 seeds，GridWorld，成功率协议 | `her_main/run_all.sh`、`her_main/results/` |
| `halfcheetah_sac/` | HalfCheetah-v5 SAC 3 seeds × 1M steps，final mean 10917（10.2k–11.4k），连续控制 benchmark | `halfcheetah_sac/README.md`、`halfcheetah_sac/results/` |

## ablations — 消融实验

| 目录 | 内容 | 入口 |
| --- | --- | --- |
| `her_ratio/` | HER ratio ∈ {0, 0.2, 0.5, 0.8, 1.0}，10 seeds；有 HER 优于无 HER，ratio=1.0 一定失败，0.2/0.5/0.8 不可分 | `her_ratio/README.md`、`her_ratio/results/` |
| `her_strategy/` | 采样策略 future / final / episode，10 seeds；episode 略好但含平凡目标 | `her_strategy/README.md`、`her_strategy/results/` |
| `sac_hparams/` | HalfCheetah SAC 超参单因素扫描（lr / batch / gamma / 初始 alpha），100k × 3 seeds；lr 1e-4 与 γ 0.95 明显更慢，init alpha 1.0 只造成约 20k 步延迟，batch / lr 1e-3 分不开 | `sac_hparams/README.md`、`sac_hparams/经验总结.md` |
