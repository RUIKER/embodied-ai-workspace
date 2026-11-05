# PPO on CartPole

在 `CartPole-v1` 上使用离散动作 PPO。

## 运行

```bash
python -m pip install -r requirements.txt
python main.py
```

策略网络输出 `Categorical` 分布，PPO 使用 clipped surrogate objective 和 GAE 更新 actor，使用 MSE loss 更新 critic。

`practice.py` 是原始混合示例，正式入口是 `main.py`。
