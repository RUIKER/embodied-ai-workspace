# TRPO 实验

本目录包含 TRPO 在离散动作和连续动作环境中的两个独立实现。

## 目录

| 目录 | 环境 | 策略分布 | 入口 |
| --- | --- | --- | --- |
| `CartPole` | `CartPole-v1` | `Categorical` | `CartPole/main.py` |
| `Pendulum` | `Pendulum-v1` | `Normal` | `Pendulum/main.py` |

两个 case 分别维护自己的配置、网络、agent、训练器和依赖文件，避免把离散动作和连续动作逻辑混在一起。

## 运行车杆

```bash
cd CartPole
python -m pip install -r requirements.txt
python main.py
```

## 运行倒立摆

```bash
cd Pendulum
python -m pip install -r requirements.txt
python main.py
```

## 算法差异

TRPO 的公共部分包括：GAE、策略 surrogate objective、KL 约束、Hessian-vector product、共轭梯度和线搜索。

- CartPole：策略输出离散动作概率，使用 `Categorical` 分布。
- Pendulum：策略输出连续动作高斯分布的均值和标准差，使用 `Normal` 分布。

不能只修改环境名称来切换两个 case，因为动作空间和策略分布不同。
