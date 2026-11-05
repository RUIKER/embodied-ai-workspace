# 具身智能实习工作区

把求职准备期的三类材料放在同一个仓库里：计划文档、强化学习 / 模仿学习代码、工业知识库 Search Agent。
用途是让多台机器上的开发工具（Claude Code、Codex 等）直接打开一个工作区就能读写全部内容。

## 目录

| 目录 | 内容 | 原仓库 |
| --- | --- | --- |
| `plan/` | 作战计划、周部署清单、公司表、JD 对照表、每周反思、进度追踪 | — |
| `hands-on-reinforcement-learning/` | 01–14 章：DQN / 策略梯度 / DDPG / TD3 / SAC / HER / 模仿学习 / 经典控制对照；统一评估协议的对照实验 | [RUIKER/hands-on-reinforcement-learning](https://github.com/RUIKER/hands-on-reinforcement-learning) |
| `industrial-search-agent/` | 工业知识库 RAG：语料清洗切分、向量索引、BM25 / 向量混合检索、rerank、MCP 工具封装 | `RUIKER/industrial-search-agent`（私有） |

## 不在仓库里的东西

| 内容 | 原因 |
| --- | --- |
| `**/.venv/` | 虚拟环境体积大，克隆后按各项目 README 重建 |
| `industrial-search-agent/data/`、`docs/materials/` | 真实语料与标准答案属于内部文件，只在本地流转 |
| `plan/` 下的简历 docx、`plan/_backup/` | 含手机号和邮箱，不公开 |

## 从零跑起来

```bash
# 强化学习仓库
cd hands-on-reinforcement-learning
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -I 14_Control/Pendulum/evaluate.py --controller both --compare-sac

# 工业知识库 Search Agent（语料需自行放回 data/）
cd industrial-search-agent
pip install -r requirements.txt
python scripts/smoke_test.py        # 用虚构语料跑通全链路
```

## 说明

- 两个子项目原本是独立仓库，这个工作区是它们的协作快照。原仓库仍在，提交历史保留在那里。
- 各子项目的评估口径、边界和已知限制写在各自的 `README.md` 里，引用数字前先看那几节。
- 新增文件时注意根目录 `.gitignore` 的机密项：`data/`、`docs/materials/`、简历 docx 不进版本库。
