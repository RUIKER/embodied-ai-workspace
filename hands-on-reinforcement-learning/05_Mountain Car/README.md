# Mountain Car Actor-Critic

## Pipeline

1. `config.py`: 集中管理实验参数。
2. `environment.py`: 封装 Gymnasium 的 reset、step 和 seed 逻辑。
3. `features.py`: 标准化状态，并生成多组 RBF 特征。
4. `estimators.py`: 实现线性 Gaussian actor 与线性 value critic。
5. `trainer.py`: 执行 TD actor-critic 训练并记录统计数据。
6. `visualization.py`: 绘制奖励和步数曲线。
7. `main.py`: 组装并运行完整 pipeline。

运行：

```bash
python3 -m pip install -r requirements.txt
python3 main.py
```

原来的 `code.py` 保留为兼容入口，也可以直接运行：

```bash
python3 code.py
```
