"""
第2周:RAG 完整链路。

串联:用户问题 → retriever 检索 → generator 生成 → 返回 {answer, sources, retrieved}。
"""
from __future__ import annotations
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.retriever import retrieve
from rag.generator import generate
from config import FINAL_TOP_K


def answer(question: str, mode: str = "rerank", top_k: int = FINAL_TOP_K,
           with_context: bool = True) -> dict:
    """
    完整 RAG 问答。

    返回:
      {
        "question": str,
        "answer": str,
        "sources": [{id, source, category, score}, ...],  # 引用来源
        "retrieved": [...],  # 完整检索片段
        "mode": str,
        "latency_ms": float,  # 端到端延迟
      }
    """
    t0 = time.time()
    retrieved = retrieve(question, top_k=top_k, mode=mode)
    ans = generate(question, retrieved) if with_context else generate(question, retrieved)
    latency = (time.time() - t0) * 1000

    sources = [{
        "id": r.get("id"),
        "source": r.get("source"),
        "category": r.get("category"),
        "score": round(r.get("rerank_score", r.get("score", 0.0)), 4),
    } for r in retrieved]

    return {
        "question": question,
        "answer": ans,
        "sources": sources,
        "retrieved": retrieved,
        "mode": mode,
        "latency_ms": round(latency, 1),
    }


if __name__ == "__main__":
    q = "常州市在减速器领域有哪些代表性龙头企业?"
    res = answer(q)
    print("问题:", res["question"])
    print("模式:", res["mode"], "| 延迟:", res["latency_ms"], "ms")
    print("\n回答:\n", res["answer"])
    print("\n来源:")
    for s in res["sources"]:
        print("  -", s)
