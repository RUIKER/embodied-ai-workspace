# AGENTS.md

多项目工作区。开始工作前先读这一页，再读对应子项目的 `README.md`。

## 目录

- `plan/`：求职计划与文档。改动前先看 `00_作战计划.docx`，那是当前版本的唯一事实来源。
- `hands-on-reinforcement-learning/`：RL 算法章节 + `experiments/` 对照实验 + `14_Control/Pendulum` 经典控制对照。
- `industrial-search-agent/`：RAG / MCP 项目。

## 硬性规则

1. **不要提交机密文件**：`industrial-search-agent/data/`、`industrial-search-agent/docs/materials/`、`plan/` 下的简历 docx、`plan/_backup/`、任何 `.venv/`。根目录 `.gitignore` 已经覆盖，不要用 `git add -f` 绕过。
2. **不要运行大规模实验**（多 seed 并行、超参扫描、长时间训练）。需要时把命令、预计耗时和落盘位置写进计划，由人决定何时启动。
3. **评估数字必须可复现**：引用任何数字前，确认对应的评估协议和 seed 集写在该项目的 README 里。
4. **提交信息**用中文，格式参照现有提交：`feat(13_Imitation): ...`、`docs(14_Control): ...`。
5. 改动子项目后，至少跑一次该项目 README 里列出的 smoke / 单元检查。
6. **沉淀每日追问**：每天将用户在学习或实践中追问的问题、准确解释和相关实践证据记录到对应子项目的 `经验总结.md`。按日期和主题归档；文件不存在时创建，并在子项目 README 中添加入口。尚未验证的判断标明状态，不把 agent 的验证写成本人已完成的操作。
7. **明确当天任务和完成状态**：带本人学习时，以当前计划及实际进度区分当天任务和后续任务；提前涉及后续内容时明确提示其原定时间和定位。当日任务完成时主动告知；若只完成主要内容，清楚指出剩余复盘、用时记录或未验证事项。分别记录本人学习和 Agent 交付，不强制延长当天安排。

## 常用入口

```bash
# RL：五算法对照（已有结果，重跑约 5 分钟）
cd hands-on-reinforcement-learning/experiments/seed_experiments/five_algorithms && python -I run_all.py

# RL：经典控制对照
cd hands-on-reinforcement-learning && python -I 14_Control/Pendulum/evaluate.py --controller both

# Search Agent：虚构语料端到端
cd industrial-search-agent && python scripts/smoke_test.py
```

## 事实来源

- 计划与优先级：`plan/00_作战计划.docx`
- 本周安排：`plan/01_本周部署清单_W2.docx`
- 进度：`plan/10_进度追踪.xlsx`
- 评估口径与边界：各子项目 `README.md`
