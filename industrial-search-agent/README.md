# 工业知识库 Search Agent

面向工业零部件 / 政策 / 企业等复杂语义场景的知识库问答系统：把原有 FAQ 关键词问答升级为 **Hybrid 检索 + Rerank + 大模型生成 + MCP 工具调用** 的 RAG Agent，并规划了向 **RL 驱动的 Search Agent**（Search-R1 范式）演进的技术路线。

> 一句话定位：**工业知识库 + RAG + Agent 工具调用系统优化**。

## 技术栈

- **RAG**：LangChain + Chroma 向量库 + `BAAI/bge-small-zh-v1.5` embedding
- **Hybrid Search**：向量检索 + BM25 关键词检索加权融合（工业术语兜底）
- **Rerank**：`BAAI/bge-reranker-base` cross-encoder 重排
- **生成**：智谱 GLM API（`glm-4.6`），未配置 Key 时自动降级为返回检索片段
- **Agent / MCP**：Streamable HTTP 协议工具调用（`mcp` SDK，Python 3.9 回退 FastAPI）
- **Demo**：Gradio Web 界面

## 仓库结构

```
industrial-search-agent/
├── config.py                  # 集中配置：路径、模型、切分/检索参数
├── app.py                     # Gradio Web demo
├── requirements.txt
├── knowledge/                 # 知识结构化与索引
│   ├── schema.py              # 三层结构 + 八大类标签
│   ├── ingest.py              # 文档加载 → 清洗 → 切分 → 打标签
│   └── build_index.py         # 构建 Chroma 向量索引
├── rag/                       # RAG 链路
│   ├── embeddings.py          # embedding / reranker 单例加载
│   ├── retriever.py           # 四档检索：bm25 / vector / hybrid / rerank
│   ├── generator.py           # LLM 生成
│   └── chain.py               # 检索 → 生成 全链路编排
├── mcp_server/server.py       # MCP 工具封装（3 个工具）
├── eval/                      # 评估
│   ├── answers.py             # 标准答案适配（docx 表格 / CSV / JSON → answers.json）
│   ├── run_eval.py            # 100 题四档检索对比（默认不调外部 API）
│   └── metrics.py             # 命中率 / 关键点覆盖率 / 相关性 / 拒答率
├── scripts/smoke_test.py      # 虚构语料端到端冒烟（ingest → 索引 → 评估）
├── data/
│   ├── raw/                   # 原始知识库文档（txt/md/docx/pdf，本地放置）
│   ├── processed/chunks.jsonl # 切分后的 chunk（带标签 + 来源）
│   └── questions/test_questions.json  # 100 个测试问题
└── docs/
    ├── architecture.md        # 三张 Mermaid 架构图
    ├── getting-started.md     # 1 小时上手导读
    ├── roadmap-search-r1-rl.md# 后续路线：RAG → RL Search Agent
    └── materials/             # 项目相关文档（知识库目录、任务规划）
```

> `data/chroma_db/` 为本地生成的向量索引，不入库；放入真实文档后重跑 `build_index.py` 即可重建。

## 快速开始

```bash
pip install -r requirements.txt

# 1. 把原始知识库文档放入 data/raw/（txt/md/docx/pdf）
# 2. 清洗切分 → data/processed/chunks.jsonl
python knowledge/ingest.py

# 3. 构建向量索引 → data/chroma_db/（首次经 modelscope 自动下载 bge 模型）
python knowledge/build_index.py

# 4. Web demo
python app.py

# 5. MCP Server（供 Agent 调用）
python mcp_server/server.py

# 6. 标准答案适配 → data/private/answers.json（docx 表格 / CSV / JSON 三选一）
python -m eval.answers 标准答案.docx
python -m eval.answers 标准答案.docx --dry-run   # 只识别格式，不写文件

# 7. 四档检索对比（默认只算检索侧指标，不调用外部 API）
python eval/run_eval.py             # 全量
python eval/run_eval.py --limit 10  # 调试：只跑前 10 题
python eval/run_eval.py --gen       # 启用生成与相关性判分（会把片段发给 GLM）

# 8. 没有真实语料时，先用虚构语料验证链路
python scripts/smoke_test.py
```

> `data/private/`（标准答案）、`data/eval_results.jsonl`（逐题检索原文）、`data/smoke/` 均在 `.gitignore` 中。

## 环境变量

- `ZHIPUAI_API_KEY`：智谱 GLM API Key，生成环节使用；未配置时自动降级为"返回检索片段"，链路仍可演示。

## 检索模式（四档，可对比）

| mode | 说明 | 用途 |
|------|------|------|
| `bm25` | 纯关键词 | 模拟原系统基线 |
| `vector` | 纯向量 | RAG 基线 |
| `hybrid` | 向量 + BM25 加权融合（0.6 / 0.4） | 工业术语兜底 |
| `rerank` | hybrid + cross-encoder 重排 | **最终方案（默认）** |

## MCP 工具

| 工具 | 作用 |
|------|------|
| `industrial_knowledge_search` | 工业知识库检索（支持 mode / top_k 参数） |
| `enterprise_query` | 企业信息查询 |
| `component_param` | 零部件参数查询 |

## 路线图：从静态 RAG 到 RL Search Agent

当前系统的检索策略是**写死的规则**（单轮、固定流程）。后续方向是借鉴 [Search-R1](https://github.com/PeterGriffinJin/Search-R1)，让模型通过强化学习**自主决定何时检索、搜什么、搜几轮、何时作答**，把检索器视为冻结的环境，仅训练模型的决策与生成策略。

分阶段路线（详见 [`docs/roadmap-search-r1-rl.md`](docs/roadmap-search-r1-rl.md)）：

1. **阶段 A｜可训练化改造**：实现多轮交错 `reason-search` rollout 与标签解析，先以 prompt 版多轮 Agent 拿到 zero-RL 基线
2. **阶段 B｜数据与奖励**：扩到 300~500 题训练集，先照搬论文纯 EM 结果奖励，实现检索 token loss-mask
3. **阶段 C｜跑通 RL**：Qwen2.5-3B + PPO（显存受限退化 GRPO），检索器复用现有 Hybrid + Rerank
4. **阶段 D｜多工具动作空间**：`<search>` / `<enterprise_query>` / `<component_param>` 工具选择
5. **阶段 E｜评估与工程化**：基线 / 静态 RAG / Prompt 多轮 / RL Agent 四方对比 + 消融

## 文档索引

| 文档 | 内容 |
|------|------|
| [docs/getting-started.md](docs/getting-started.md) | 逐模块导读、运行命令速查、动手练习 |
| [docs/architecture.md](docs/architecture.md) | 技术架构 / RAG 流程 / MCP 调用三张图 |
| [docs/roadmap-search-r1-rl.md](docs/roadmap-search-r1-rl.md) | RAG → RL Search Agent 的完整实施路线与风险取舍 |
| [docs/materials/](docs/materials/) | 知识库目录与测试问题、任务规划等原始材料 |

## 进度

| 模块 | 状态 |
|------|------|
| 知识结构化、清洗切分、100 题 | 代码完成 |
| Chroma 索引、四档检索、rerank | 代码完成 |
| 生成（GLM） | 代码完成，默认关闭 |
| MCP Server 三个工具 | 代码完成 |
| 标准答案适配（docx / csv / json） | 代码完成 |
| 有真实语料的四档对比表 | **阻塞中**：已有 100 题（`data/questions/test_questions.json`），缺原始知识库正文 + 标准答案 |
| 链路冒烟基线 | 已完成：`data/smoke/eval_summary.json`，10 题四档检索命中率 100%（虚构语料过简单，仅验证链路） |
| 多轮 Agentic 检索 | 未开始（第 3 周） |
| RL 检索决策训练 | 未开始，见路线图 |

## 基线数字（截至 2026-11-04）

**Smoke 基线**（`python scripts/smoke_test.py`）：

| 方法 | 命中率 | Top1 命中率 | 平均延迟 |
|---|---:|---:|---:|
| BM25 | 100% | 100% | 0.1 ms |
| 向量检索 | 100% | 100% | 9.1 ms |
| 向量+BM25 融合 | 100% | 100% | 10.9 ms |
| 融合+rerank | 100% | 100% | 737.6 ms |

> 说明：虚构语料的答案与问题高度对应，四档都能 100% 命中；这个数字只证明检索链路可跑通，
> 不能证明真实任务上的性能。真实基线需要在 `data/raw/` 放入政府文件正文、并生成标准答案后重跑。


已知限制：`enterprise_query` 与 `component_param` 目前都只是对同一次检索的包装，不是独立数据源；
生成环节未配置 `ZHIPUAI_API_KEY` 时会降级为返回 Top-1 片段。
