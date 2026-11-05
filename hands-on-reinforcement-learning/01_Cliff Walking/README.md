# Cliff Walking

在手写 Cliff Walking 环境上比较表格型 Q-learning 和 SARSA。

## 文件说明

- `environment.py`：Cliff Walking 环境。
- `q_learning.py`：Q-learning 的 Q 表和更新逻辑。
- `sarsa.py`：SARSA 的 Q 表和更新逻辑。
- `train.py`：训练入口。
- `config.py`：地图、学习率、折扣因子和探索率配置。

## 运行

该项目只依赖 Python 标准库：

```bash
python train.py
```

## 切换算法

在 `train.py` 中选择：

```python
from q_learning import QTable as AlgorithmTable
```

或者切换为：

```python
from sarsa import SarsaTable as AlgorithmTable
```

Q-learning 使用下一状态的最大 Q 值作为目标；SARSA 使用实际选择的下一动作对应的 Q 值作为目标。
