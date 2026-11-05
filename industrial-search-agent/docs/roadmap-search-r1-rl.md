# 工业 Search Agent 后续完善路线：从静态 RAG 到 RL 驱动的搜索智能体（Search-R1 魔改）

> 目标读者：本人（后续继续迭代该实习项目）
> 核心命题：把当前"检索—拼接—生成"的**静态 RAG** 升级为**用强化学习训练检索决策策略**的 Search Agent，使简历中的 RL 表述有真实实现支撑，而非包装。

---

## 0. 一句话定位

当前系统 = `Hybrid 检索 + Reranker + GLM 生成 + MCP 工具封装`（检索策略是**写死的规则**）。
升级目标 = 让模型**自己学会**"要不要搜、搜什么、搜几轮、何时停下作答"——这正是 **Search-R1** 解决的问题。

---

## 1. Search-R1 是什么（原理速记）

Search-R1（Jin et al., 2025，*Training LLMs to Reason and Leverage Search Engines with RL*）的核心思想：

1. **交错式推理-检索（interleaved reason-and-search）**
   模型在一段生成里自由穿插思考与检索，结构化为特殊 token：
   ```
   <think> 需要先确认谐波减速器的国产化率… </think>
   <search> 谐波减速器 国产 龙头企业 </search>
   <information> [检索系统返回的 top-k 片段] </information>
   <think> 结合来料，答案应为… </think>
   <answer> …… </answer>
   ```
   模型自主决定 `<search>` 的**时机、次数、query 内容**，可多轮。rollout 循环见论文 Algorithm 1：逐 token 生成，遇到 `</search>` 就抽取 query、调检索器、把结果包进 `<information>…</information>` 回填继续；遇到 `</answer>` 则结束；**若一轮既没检索也没作答（非法动作），系统追加一句 "My action is not correct. Let me rethink." 让模型重想**；直到触发 `<answer>` 或达到最大动作预算 B。

2. **用 RL 直接优化最终答案**
   - 算法：论文在 PPO 与 GRPO 上都验证过，**默认用 PPO**。实测结论（务必记牢，别写反）：**GRPO 收敛更快，但训练多步后会出现 reward collapse（奖励崩溃）；PPO 更稳定**（PPO 靠 critic 需 warm-up 所以慢），两者最终 reward 相当 —— 因此论文选 PPO 作默认。GRPO 的优点是无需 critic、省显存，适合资源受限先跑通，但要注意加 early-stop 防崩溃。
   - 奖励：论文用**极简的结果奖励（outcome reward）= EM 精确匹配**（`r = EM(a_pred, a_gold)`）。**关键：论文明确"不加 format reward、不训神经奖励模型"**，因为模型在 RL 中自然就学会遵守标签格式，且神经奖励模型易被 reward hacking、成本高。
   - **检索 token 损失屏蔽（retrieved token masking）**：`<information>` 内是环境返回、非模型生成，计算 policy loss 与 KL loss 时都屏蔽掉，避免模型去"背检索内容"。论文消融证实其关键：Qwen2.5-7B 上带 mask 0.431 vs 不带 0.343。

3. **检索器视为环境的一部分**
   训练时检索器参数**冻结**，只训练 LLM 的**决策+生成策略**。论文用 E5 检索 + 2018 Wikipedia 语料、每次固定返回 **3 篇** passage；本项目对应替换为现有 Hybrid + Reranker 工业检索链路。

4. **论文实测的训练动态（可写进答辩，显示读懂了）**
   - 有效检索次数**随训练上升**：模型逐渐学会"多搜几次"（Fig.2d）。
   - 回复长度呈**"降—升—稳"**：前期砍掉废话、后期因频繁检索变长（Fig.2c）。
   - base 与 instruct 模型**最终性能接近**，instruct 只是收敛更快、起点更高 —— base 模型也能训得起来。
   - 规模效应：7B 相对 RAG 提升 24%、3B 提升 20%，越大越会"学搜索"。

相关工作可一并了解：Search-o1（prompt 式 agentic 检索，论文里作为 baseline）、R1-Searcher、ReSearch、DeepRetrieval（都在"用 RL 学检索/查询改写"这条线上）。

---

## 2. 魔改点：把 Search-R1 落到工业知识库

Search-R1 原论文面向开放域 QA（用 Wikipedia + 通用检索）。工业场景要做的"魔改"：

| 原始 Search-R1 | 工业魔改版 |
|---|---|
| 通用检索器（e5 / BM25 over Wiki） | 复用现有 **BGE + BM25 Hybrid + BGE-Reranker** 工业检索链路 |
| 单一维基语料 | **多源工具**：知识库检索 / 企业查询 / 零部件参数 / 政策检索（已封装为 MCP tool） |
| 单一 `<search>` 动作 | **多工具动作空间**：`<search>` / `<enterprise_query>` / `<component_param>`，模型学会**选哪个工具** |
| EM 对齐短答案（论文原做法，无 format reward） | 工业答案偏长含术语，EM 过严 → **改良假设**：奖励改用 F1/术语命中，或 LLM-as-judge（⚠️ 偏离论文，属本项目待验证设计，非 Search-R1 原方案） |
| Qwen2.5-3B/7B（base 或 instruct 均可训） | 资源受限 → 先用 **3B**（论文验证过的最小规模）；1.5B 论文未验证，属推测，跑不通再降 |

> 一句话：**检索器不变，把"何时调用哪个工具、query 怎么写、搜几轮"交给 RL 学**——这就是本项目 RL 化的核心增量。
>
> ⚠️ **忠于论文提示**：Search-R1 原方案刻意保持"极简奖励（纯 EM）+ 不加 format reward + 不训奖励模型"。本文档在工业场景下提出的 F1/judge/format 惩罚等都属**改良假设**，答辩时应说清"这是我基于工业长答案特点的改动，论文本身用的是纯 EM"，不要说成论文做法。

---

## 3. 分阶段实施路线（建议按序推进，每阶段都可独立写进简历/答辩）

### 阶段 A：可训练化改造（把现有系统变成"能被 RL 调用的环境"）
目标：不引入 RL，先把链路改成**多轮、可交错、可 rollout** 的形态。

1. **定义交互协议**：实现一个 `rollout(question) -> trajectory` 函数，支持模型输出中解析 `<think>/<search>/<answer>` 标签，遇到 `<search>` 就调现有 `rag/retriever.py`，把结果包进 `<information>` 回填，循环直到 `<answer>` 或达到最大轮数。
   - 复用：`rag/retriever.py`（hybrid + rerank）、`mcp_server/server.py`（三个工具）。
   - 新增：`agent/rollout.py`（多轮循环 + 标签解析 + 停止条件）。
2. **先用 Prompt 版验证可行性（zero-RL 基线）**：用 few-shot prompt 让 GLM/Qwen 按上述格式**多轮自主检索**（即 "Search-o1 / ReAct" 式，不训练）。这一步能立刻拿到"多轮 agentic RAG"的 demo 和指标，作为 RL 的对照基线。

### 阶段 B：数据与奖励（RL 的前提）
1. **构造训练问题集**：在现有 `data/questions/test_questions.json`（100 题）基础上扩到 **300~500 题**，每题标注 `gold_answer` 或 `gold_支撑片段`。可用 GLM 对知识库自动"造问答对" + 人工校验。
2. **设计奖励函数** `rl/reward.py`：
   - **先照搬论文的纯 EM 结果奖励跑通基线**（`r = EM(a_pred, a_gold)`，不加 format reward）——这是最忠于 Search-R1、最不易出错的起点。
   - 再做工业改良（⚠️ 偏离论文，需单独消融验证是否更好）：`r_answer` 用 F1/术语命中缓解 EM 对长答案过严；可选 `r_judge`（LLM-as-judge 0/1）。**format 惩罚谨慎加**——论文证明不加也能学会格式，加了反而可能引入 reward hacking。
3. **检索 token masking**：训练样本里标记 `<information>…</information>` 区间为 loss-mask（policy loss 与 KL loss 都要 mask）。论文消融证明这是稳定训练的关键，务必实现。

### 阶段 C：跑通 RL 训练（核心里程碑）
1. **框架选型**：**首选直接 fork [Search-R1 官方仓库](https://github.com/PeterGriffinJin/Search-R1)**（基于 veRL，已实现 rollout+masking+PPO/GRPO，省去自己踩坑）；备选 TRL(GRPOTrainer) / OpenRLHF。
2. **算法**：**默认 PPO**（论文默认、更稳）。若显存吃紧改用 GRPO（无 critic、省显存），但**须加 early-stop 或监控 reward，防止论文报告的 GRPO 后期崩溃**。
3. **模型**：`Qwen2.5-3B`（base 或 instruct，论文验证过的最小规模）起步，单卡 24G 可 LoRA。1.5B 属未验证推测。
4. **最小闭环**：检索器=现有 Hybrid+Rerank（冻结），策略模型=Qwen-3B，奖励=先纯 EM，算法=PPO，先在 100~300 题上过拟合验证"reward 能涨、有效检索次数上升"。
5. **产出**：训练曲线（reward / 有效检索次数 / 回复长度随 step 变化，对齐论文 Fig.2）、若干 rollout 轨迹可视化。

### 阶段 D：多工具与工业增强（真正的"魔改亮点"）
1. **多工具动作空间**：把 `<search>` 扩展为 `<search>/<enterprise_query>/<component_param>`，让模型学**工具选择**，奖励里加"是否调用了正确工具"。
2. **Query rewriting 作为可学动作**：模型在 `<search>` 里写的 query 本身就是被 RL 优化的对象 —— 对工业术语场景收益明显（对齐 DeepRetrieval 思路）。
3. **鲁棒性**：注入含噪/口语化问题（模拟展厅真实提问），训练策略对模糊 query 的多轮澄清-检索能力。

### 阶段 E：评估与工程化（简历量化指标来源）
1. **对比实验矩阵**（务必量化，直接进简历/答辩）：

   | 方法 | 检索轮数 | 答案 F1 | Top-k 命中 | 延迟 |
   |---|---|---|---|---|
   | 关键词基线(BM25) | 1(固定) | | | |
   | 静态 RAG(hybrid+rerank) | 1(固定) | | | |
   | Prompt 多轮 Agent(阶段A) | 自适应 | | | |
   | **RL Search Agent(GRPO, 本项目)** | **自适应** | | | |

2. **消融**：去掉 retrieved-token-masking / 去掉多工具 / 换 PPO↔GRPO，各看 reward 与 F1 变化。
3. 部署：训练好的策略模型接回 `app.py` 与 MCP，形成"可交互 demo + 训练报告"双产出。

---

## 4. 目录结构增量建议

```
industrial-search-agent/
├── agent/
│   ├── rollout.py          # 多轮交错 reason-search 循环（阶段A）
│   ├── parser.py           # <think>/<search>/<answer> 标签解析
│   └── tools.py            # 统一封装检索/企业/参数/政策工具（复用 mcp_server）
├── rl/
│   ├── reward.py           # 奖励函数（F1 + judge + format + token-mask）
│   ├── train_grpo.py       # GRPO 训练入口（veRL/TRL）
│   ├── dataset.py          # 训练问答对加载 + loss mask 标注
│   └── config_grpo.yaml    # 模型/学习率/rollout/kl 系数等
├── eval/
│   └── compare_rl.py       # 阶段E 对比实验矩阵
└── (其余复用现有 rag/ knowledge/ mcp_server/)
```

---

## 5. 关键风险与取舍

- **算力**：RL rollout 很吃显存/时间。先 LoRA + 1.5~3B + 小题集跑通，再谈规模。跑不动就以"阶段 A 的 Prompt 多轮 Agent + 完整 RL 设计文档 + 小规模训练曲线"作为交付，简历表述改为"设计并实现了基于 GRPO 的检索决策训练框架，并在 XX 题子集上验证 reward 收敛"——**真实、可答辩**。
- **奖励黑客**：模型可能学会"疯狂检索"或"编造标签"骗奖励。用 format 惩罚 + 检索次数惩罚 + token-masking 缓解。
- **数据是瓶颈**：gold 答案质量决定上限，工业问答对的构造+校验要占不少时间，尽早启动。
- **诚信底线**：简历只写**已真实跑通的部分**。若仅完成阶段 A+设计，就写"基于 Search-R1 范式设计 RL 训练方案并完成多轮 Agent 原型与小规模验证"，不要宣称完成了大规模 RL 训练。

---

## 6. 简历表述（可直接用，随完成度三选一）

- **已完成完整 RL 训练**：
  > 借鉴 Search-R1，基于 GRPO 强化学习训练模型在多轮"推理—检索"轨迹中自主决策检索时机与 query，以答案正确率为结果奖励端到端优化，工业问答 F1 较静态 RAG 提升 XX%。
- **完成原型+小规模验证（推荐，稳妥）**：
  > 基于 Search-R1 范式设计并实现 RL 检索决策训练框架（GRPO + 结果奖励 + 检索 token 屏蔽），构建多轮交错 reason-search Agent 原型，并在子集上验证 reward 收敛与检索行为涌现。
- **仅完成 Agent 化改造**：
  > 将静态 RAG 升级为多轮自主检索 Agent（reason-search 交错 + 多工具调用），并设计了面向该系统的 GRPO 强化学习优化方案。

---

## 7. 参考（据文件夹内 search-r1.pdf 核对）

- **Search-R1**: Jin et al., *Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning*, **COLM 2025**（arXiv:2503.09516v5, 2025-08）。官方代码与 checkpoint：https://github.com/PeterGriffinJin/Search-R1（基于 veRL）。
  - 核心：交错 reason-search、检索 token masking、纯 outcome(EM) 奖励；PPO 为默认、GRPO 快但后期易崩；检索器 E5+Wiki 冻结、每次取 3 篇；7 个 QA 数据集（NQ/TriviaQA/PopQA/HotpotQA/2Wiki/Musique/Bamboogle），Qwen2.5-7B/3B 相对 RAG 提升 24%/20%。
- 相关：Search-o1（prompt 式 agentic 检索，论文 baseline）、R1-Searcher、ReSearch、DeepRetrieval。
- 算法：PPO（Schulman 2017）、GRPO（Shao 2024 / DeepSeek）；上游 DeepSeek-R1-Zero（Guo 2025）。框架：veRL、TRL（GRPOTrainer）、OpenRLHF。

