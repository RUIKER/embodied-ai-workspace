"""
第2周:Hybrid 检索 + Rerank。

流程:
  向量检索 TopK(Chroma) + BM25 关键词检索 TopK
    → 加权融合(向量 0.6 / BM25 0.4)
    → bge-reranker 重排序
    → 取最终 TopK

简历亮点:
- hybrid search:工业术语(如"谐波减速器""摆线针轮")靠 BM25 兜底,弥补纯向量在专有名词上的不足。
- rerank:cross-encoder 对 query-doc 精排,提升 Top1 准确率。

支持 mode 参数,供第4周对比实验:
  "bm25"        纯关键词(模拟原系统)
  "vector"      纯向量(RAG 基线)
  "hybrid"      向量+BM25 融合
  "rerank"      hybrid + rerank(最终方案,默认)
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
from typing import List, Optional
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    CHUNKS_PATH, CHROMA_DIR, CHROMA_COLLECTION,
    TOP_K_VECTOR, TOP_K_BM25, FINAL_TOP_K,
    VECTOR_WEIGHT, BM25_WEIGHT,
)
from rag.embeddings import get_embedding_function, get_reranker


# ===== BM25 语料懒加载(单例)=====
_bm25 = None
_corpus_chunks = None  # id -> chunk dict,供回填完整字段


def _load_corpus():
    global _corpus_chunks
    if _corpus_chunks is None:
        _corpus_chunks = {}
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    c = json.loads(line)
                    _corpus_chunks[c["id"]] = c
    return _corpus_chunks


def _get_bm25():
    global _bm25
    if _bm25 is None:
        from rank_bm25 import BM25Okapi
        corpus = _load_corpus()
        # 中文按字符切分(BM25 对中文的简单有效做法)
        ids = list(corpus.keys())
        tokenized = [list(corpus[i]["text"]) for i in ids]
        _bm25 = (BM25Okapi(tokenized), ids)
    return _bm25


def _get_collection():
    import chromadb
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(CHROMA_COLLECTION, embedding_function=get_embedding_function())


# ===== 各路检索 =====
def _vector_search(query: str, top_k: int = TOP_K_VECTOR) -> List[dict]:
    """向量检索,返回带 score 的 chunk 列表(score 越高越相似)。"""
    collection = _get_collection()
    res = collection.query(query_texts=[query], n_results=top_k)
    out = []
    ids = res["ids"][0]
    dists = res["distances"][0]
    metas = res["metadatas"][0]
    docs = res["documents"][0]
    corpus = _load_corpus()
    for cid, dist, meta, doc in zip(ids, dists, metas, docs):
        # Chroma 返回的是距离(越小越相似),转成相似度分数
        score = 1.0 - dist
        c = corpus.get(cid, {"id": cid, "text": doc, **meta})
        out.append({**c, "score": score})
    return out


def _bm25_search(query: str, top_k: int = TOP_K_BM25) -> List[dict]:
    """BM25 关键词检索。"""
    bm25, ids = _get_bm25()
    scores = bm25.get_scores(list(query))
    ranked = sorted(zip(ids, scores), key=lambda x: x[1], reverse=True)[:top_k]
    corpus = _load_corpus()
    out = []
    for cid, sc in ranked:
        c = corpus[cid]
        out.append({**c, "score": float(sc)})
    return out


def _normalize(scores: List[float]) -> List[float]:
    """min-max 归一化到 [0,1],便于两路加权融合。"""
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-9:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def _fuse(vec_results, bm25_results) -> List[dict]:
    """加权融合两路结果。"""
    merged = {}
    for r in vec_results:
        merged.setdefault(r["id"], {**r}).setdefault("_vec", 0.0)
        merged[r["id"]]["_vec"] = r["score"]
    for r in bm25_results:
        merged.setdefault(r["id"], {**r}).setdefault("_bm", 0.0)
        merged[r["id"]]["_bm"] = r["score"]

    vec_scores = _normalize([v.get("_vec", 0.0) for v in merged.values()])
    bm_scores = _normalize([v.get("_bm", 0.0) for v in merged.values()])
    for v, vs, bs in zip(merged.values(), vec_scores, bm_scores):
        v["score"] = VECTOR_WEIGHT * vs + BM25_WEIGHT * bs
    return list(merged.values())


def _rerank(query: str, candidates: List[dict], top_k: int = FINAL_TOP_K) -> List[dict]:
    """bge-reranker cross-encoder 重排序。"""
    if not candidates:
        return []
    reranker = get_reranker()
    pairs = [[query, c["text"]] for c in candidates]
    scores = reranker.predict(pairs)
    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]


# ===== 统一入口 =====
def retrieve(query: str, top_k: int = FINAL_TOP_K,
             mode: str = "rerank",
             category: Optional[str] = None) -> List[dict]:
    """
    检索入口。

    mode:
      "bm25"    纯关键词(原系统基线)
      "vector"  纯向量(RAG 基线)
      "hybrid"  向量+BM25 融合
      "rerank"  hybrid + rerank(默认,最终方案)
    category: 可选,按八大类过滤(向量检索用 where)。
    返回: [{id, text, category, layer, source, score, ...}, ...]
    """
    if mode == "bm25":
        results = _bm25_search(query, top_k=top_k)
        return results[:top_k]

    if mode == "vector":
        results = _vector_search(query, top_k=top_k)
        return results[:top_k]

    # hybrid / rerank
    vec_results = _vector_search(query)
    bm25_results = _bm25_search(query)
    fused = _fuse(vec_results, bm25_results)
    fused.sort(key=lambda x: x["score"], reverse=True)

    if mode == "hybrid":
        return fused[:top_k]

    # rerank:对融合后候选(取较多候选送 rerank 更稳)
    candidates = fused[:max(top_k * 4, 20)]
    return _rerank(query, candidates, top_k=top_k)


if __name__ == "__main__":
    # demo:用一个问题测试四档检索
    q = "谐波减速器是什么?目前被哪个国家垄断?"
    for m in ["bm25", "vector", "hybrid", "rerank"]:
        print(f"\n===== mode={m} =====")
        for r in retrieve(q, mode=m, top_k=3):
            print(f"  [{r['score']:.4f}] {r['category']} | {r['text'][:50]}")
