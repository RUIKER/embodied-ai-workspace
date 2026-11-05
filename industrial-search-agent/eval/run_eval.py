"""
四档检索评估：bm25 / vector / hybrid / rerank。

指标（只算检索侧，默认不调用任何外部 API）：
  命中率     标准答案的关键字是否出现在 Top-K 片段里
  关键点覆盖率 标准答案有多少比例的字出现在 Top-K 片段里（比命中率细一档）
  延迟       单题端到端耗时

生成环节默认关闭。打开后（--gen）会把检索片段和标准答案发给智谱 GLM，
这一步是否允许由你来决定——语料属于政府文件，默认不外发。

标准答案从 data/private/answers.json 读（由 eval/answers.py 生成）。
没有标准答案时无法判定命中率，脚本会直接退出，而不是把 null 当 0 算。

输出：
  data/eval_results.jsonl  每题每个模式的明细（含检索片段，已在 .gitignore 中排除）
  data/eval_summary.json   汇总表
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DATA_DIR, FINAL_TOP_K, QUESTIONS_PATH
from rag.retriever import retrieve
from eval.metrics import (hit_rate, keypoint_coverage, first_hit_rank,
                          relevance, is_refusal)
from eval.answers import DEFAULT_OUT as ANSWERS_PATH, load_saved, coverage

EVAL_DIR = Path(os.environ.get("KB_EVAL_DIR", DATA_DIR))
EVAL_RESULTS_PATH = EVAL_DIR / "eval_results.jsonl"
EVAL_SUMMARY_PATH = EVAL_DIR / "eval_summary.json"

MODES = ["bm25", "vector", "hybrid", "rerank"]
LABELS = {
    "bm25": "BM25 关键词",
    "vector": "向量检索",
    "hybrid": "向量+BM25 融合",
    "rerank": "融合+rerank",
}


def load_questions():
    return json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))


def attach_answers(questions, answers):
    """把标准答案挂到题目上；返回 (带答案的题, 无答案的题)。"""
    out, missing = [], []
    for q in questions:
        gold = answers.get(int(q["id"]))
        item = {**q, "gold": gold}
        (out if gold else missing).append(item)
    return out, missing


def run_single(question: str, gold: str, mode: str, do_generate: bool = False) -> dict:
    t0 = time.time()
    retrieved = retrieve(question, top_k=FINAL_TOP_K, mode=mode)
    answer = None
    if do_generate:
        from rag.generator import generate
        answer = generate(question, retrieved)
    latency = (time.time() - t0) * 1000
    return {
        "retrieved": retrieved,
        "answer": answer,
        "latency_ms": round(latency, 1),
        "hit": hit_rate(retrieved, gold),
        "coverage": round(keypoint_coverage(retrieved, gold), 3),
        "rank": first_hit_rank(retrieved, gold),
        "rel": relevance(answer, gold) if answer is not None else None,
        "refusal": is_refusal(answer) if answer is not None else None,
    }


def run_eval(limit=None, modes=None, do_generate=False, answers_path=ANSWERS_PATH):
    questions = load_questions()
    answers_file = Path(answers_path)
    if not answers_file.exists():
        raise SystemExit(
            f"找不到标准答案 {answers_file}。\n"
            f"先运行: python -m eval.answers <标准答案文件> --out {answers_file}"
        )
    answers = load_saved(answers_file)
    cov = coverage(answers, questions)
    scored, missing = attach_answers(questions, answers)
    if limit:
        scored = scored[:limit]

    print(f"题库 {cov['questions']} 题，标准答案覆盖 {cov['answered']} 题，本次评估 {len(scored)} 题")
    if missing:
        print(f"[警告] {len(missing)} 题没有标准答案，不计入命中率分母")
    if not scored:
        raise SystemExit("没有一道题有标准答案，无法评估命中率。")

    modes = modes or MODES
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    # 预热：首次调用要加载 embedding / reranker 模型（各 1 秒级），
    # 不计入延迟统计，否则第一题会把整列平均延迟拉高。
    for m in modes:
        try:
            retrieve(scored[0]["question"], top_k=FINAL_TOP_K, mode=m)
        except Exception as e:
            print(f"[警告] {m} 预热失败: {e}")
    summary = {m: {"n": 0, "hit": 0, "cov": 0.0, "top1": 0, "mrr": 0.0,
                   "rel": 0.0, "rel_n": 0, "latency": 0.0, "refusal": 0,
                   "errors": 0} for m in modes}

    with open(EVAL_RESULTS_PATH, "w", encoding="utf-8") as fp:
        for i, q in enumerate(scored, 1):
            row = {"id": q["id"], "question": q["question"], "gold": q["gold"], "modes": {}}
            for m in modes:
                try:
                    r = run_single(q["question"], q["gold"], m, do_generate=do_generate)
                except Exception as e:  # 单题失败不影响整表
                    r = {"latency_ms": 0.0, "hit": False, "coverage": 0.0,
                         "rank": None, "rel": None, "refusal": None, "retrieved": [],
                         "answer": f"[ERROR] {e}"}
                    summary[m]["errors"] += 1
                row["modes"][m] = {
                    "hit": r["hit"],
                    "coverage": r["coverage"],
                    "rank": r["rank"],
                    "rel": r["rel"],
                    "refusal": r["refusal"],
                    "latency_ms": r["latency_ms"],
                    "top_source": r["retrieved"][0].get("source") if r["retrieved"] else None,
                    "top_ids": [c.get("id") for c in r["retrieved"]],
                    "answer": r["answer"],
                }
                s = summary[m]
                s["n"] += 1
                s["hit"] += int(r["hit"])
                s["cov"] += r["coverage"]
                s["top1"] += int(r["rank"] == 1)
                s["mrr"] += (1.0 / r["rank"]) if r["rank"] else 0.0
                s["latency"] += r["latency_ms"]
                if r["rel"] is not None:
                    s["rel"] += r["rel"]
                    s["rel_n"] += 1
                    s["refusal"] += int(bool(r["refusal"]))
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 10 == 0 or i == len(scored):
                print(f"  进度 {i}/{len(scored)}")

    table = []
    for m in modes:
        s = summary[m]
        n = max(1, s["n"])
        table.append({
            "mode": m,
            "method": LABELS.get(m, m),
            "n": s["n"],
            "hit_rate": round(s["hit"] / n * 100, 1),
            "coverage": round(s["cov"] / n, 3),
            "hit_at_1": round(s["top1"] / n * 100, 1),
            "mrr": round(s["mrr"] / n, 3),
            "relevance": round(s["rel"] / s["rel_n"], 3) if s["rel_n"] else None,
            "refusal_rate": round(s["refusal"] / s["rel_n"] * 100, 1) if s["rel_n"] else None,
            "avg_latency_ms": round(s["latency"] / n, 1),
            "errors": s["errors"],
        })

    EVAL_SUMMARY_PATH.write_text(json.dumps(
        {"n_questions": len(scored), "generate": do_generate, "table": table},
        ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 78)
    print(f"四档检索对比（{len(scored)} 题，生成环节{'开' if do_generate else '关'}）")
    print("=" * 78)
    print(f"{'方法':<16}{'命中率%':>9}{'Top1%':>8}{'MRR':>8}{'覆盖率':>8}{'延迟ms':>10}")
    print("-" * 78)
    for r in table:
        print(f"{r['method']:<16}{r['hit_rate']:>9.1f}{r['hit_at_1']:>8.1f}"
              f"{r['mrr']:>8.3f}{r['coverage']:>8.3f}{r['avg_latency_ms']:>10.1f}")
    print("=" * 78)
    print(f"逐题明细: {EVAL_RESULTS_PATH}（含检索原文，勿入库）")
    print(f"汇总: {EVAL_SUMMARY_PATH}")
    return table


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="只跑前 N 题（调试）")
    p.add_argument("--modes", nargs="+", default=None, help="默认 bm25 vector hybrid rerank")
    p.add_argument("--gen", action="store_true", help="启用生成环节（会调用外部 API）")
    p.add_argument("--answers", default=str(ANSWERS_PATH))
    args = p.parse_args()
    run_eval(limit=args.limit, modes=args.modes, do_generate=args.gen,
             answers_path=args.answers)
