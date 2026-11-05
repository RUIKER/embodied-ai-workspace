# 上手指南:如何运行与熟悉本项目

> 目标:1 小时内看懂项目在做什么、能跑起来、知道每个文件的作用。

## 0. 一句话理解

**把工业文档切成小片段 → 变成向量存进库 → 用户提问时检索最相关片段 → 喂给大模型生成回答**,并把这套检索能力封装成 MCP 工具供 Agent 调用,最后用 100 道题量化效果。

## 1. 环境准备(一次性)

```bash
cd industrial-search-agent
pip install -r requirements.txt
```
模型(bge embedding + reranker)首次运行时经 modelscope 自动下载,已缓存到 `~/.cache/modelscope/`,之后离线可用。

(可选)配 GLM 生成 Key,否则生成环节自动降级为"返回检索片段":
```bash
export ZHIPUAI_API_KEY=你的key
```

## 2. 运行命令速查(从数据到效果)

```bash
# 1. 文档→知识片段(产出 data/processed/chunks.jsonl)
python knowledge/ingest.py

# 2. 片段→向量索引(产出 data/chroma_db/)
python knowledge/build_index.py

# 3. Web 问答界面(浏览器打开)
python app.py

# 4. MCP 工具服务(供 Agent 调用)
python mcp_server/server.py

# 5. 100题评估+三档对比表
python eval/run_eval.py
python eval/run_eval.py --limit 10   # 调试:只跑10题
```

## 3. 逐模块导读(按数据流顺序读)

### ① config.py — 全局配置
所有参数集中在此。**改这里就能调系统行为**,不用改业务代码。
- `CHUNK_SIZE / CHUNK_OVERLAP`:切片大小(600字)和重叠(80)
- `TOP_K_VECTOR / TOP_K_BM25 / FINAL_TOP_K`:各路检索召回数
- `VECTOR_WEIGHT / BM25_WEIGHT`:hybrid 融合权重(0.6 / 0.4)
- `EMBED_MODEL / RERANK_MODEL`:模型名

### ② knowledge/schema.py — 知识结构定义
两个分类体系:
- **三层** `Layer`:基础层(事实)/产业层(关系)/场景层(应用)
- **八大类** `Category`:对应知识库目录一~八章
- `classify_tag(text)`:用关键词规则给每段文字打标签。**想改分类规则,改 `CATEGORY_KEYWORDS`。**

### ③ knowledge/ingest.py — 文档清洗切分(第1周核心)
流程:`load_text` → `clean_text` → `split_text` → `classify_tag` → 写 jsonl
- 支持 txt/md/docx/pdf
- 优先用 LangChain 的 RecursiveCharacterTextSplitter(中文友好),没装则用 `_fallback_split`
- 输出每行:`{id, text, category, layer, source}`

### ④ knowledge/build_index.py — 建向量索引(第2周)
读 chunks → 用 bge embedding 向量化 → 存进 Chroma(持久化)。metadata 带 category/layer/source,可过滤。

### ⑤ rag/embeddings.py — 模型加载
`get_embedding_function()` 和 `get_reranker()` 是单例,走 modelscope 本地路径。**build_index 和 retriever 共用同一模型,这点很重要。**

### ⑥ rag/retriever.py — 四档检索(项目灵魂)
`retrieve(query, mode=...)`:
- `bm25`:纯关键词(模拟原系统)
- `vector`:纯向量
- `hybrid`:向量+BM25 加权融合(`_fuse`)
- `rerank`:hybrid 后再用 cross-encoder 精排(`_rerank`,默认)

关键函数:`_vector_search` / `_bm25_search` / `_fuse` / `_rerank`。**对比实验就靠这个 mode 参数。**

### ⑦ rag/generator.py — LLM 生成
`generate(question, contexts)`:拼 prompt(带来源编号)→ 调 GLM → 无 Key 降级返回 top1 片段。
`SYSTEM_PROMPT` 约束模型"只基于资料回答、不编造、附来源"。

### ⑧ rag/chain.py — 串联完整链路
`answer(question)`:检索 → 生成 → 返回 `{answer, sources, retrieved, latency_ms}`。

### ⑨ mcp_server/server.py — 工具封装(第3周)
三个工具:`industrial_knowledge_search` / `enterprise_query` / `component_param`。
有 mcp SDK(3.10+)走标准 MCP;否则 FastAPI 回退(3.9)。

### ⑩ eval/run_eval.py — 评估对比(第4周)
对 100 题跑 bm25/vector/rerank 三档,输出命中率/相关性/拒答率/延迟对比表。

## 4. 动手练习(改一处,看变化)

1. **改切片大小**:`config.py` 里 `CHUNK_SIZE=300`,重跑 ingest+build_index,看 chunk 数变多、检索粒度变细。
2. **改检索模式**:`rag/retriever.py` 末尾 demo 的 `q` 换个问题,跑 `python rag/retriever.py`,对比四档结果。
3. **改 hybrid 权重**:`VECTOR_WEIGHT=0.3, BM25_WEIGHT=0.7`,看关键词权重提高后结果变化。
4. **加分类关键词**:`schema.py` 的 `CATEGORY_KEYWORDS` 加一个企业名,重跑 ingest 看分类变化。
5. **跑评估**:`python eval/run_eval.py --limit 10`,看三档对比表。

## 5. 换成真实知识库文档

```bash
# 把真实文档(政策/企业/产业链/词库)放进 data/raw/
# 仓库内 docs/materials/ 下的 docx 仅供回顾，不要直接作为知识库输入
python knowledge/ingest.py
python knowledge/build_index.py
python eval/run_eval.py
```
真实文档有答案内容后,检索命中率和回答质量会显著提升(示例文档只是问题列表,所以效果有限)。

## 6. 项目结构与四周对应

| 周 | 文件 | 做什么 |
|---|---|---|
| 1 | schema.py, ingest.py | 知识结构化 + 清洗切分 |
| 2 | build_index.py, rag/* | 向量索引 + RAG 链路 |
| 3 | mcp_server/server.py | 工具封装 |
| 4 | eval/*, docs/architecture.md | 评估 + 架构图 |
