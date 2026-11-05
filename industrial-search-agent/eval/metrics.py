"""
检索与回答的评估指标。

全部基于字符级重叠，不调用任何外部服务——语料是政府文件，评测默认不外发。

指标：
- hit_rate            标准答案的有区分度字符里，≥40% 出现在 Top-K 任一片段 → 命中
- keypoint_coverage   这些字符被 Top-K 覆盖的比例（[0,1]，按题平均）
- first_hit_rank      第一个命中片段的名次（1 起）；没有命中返回 None
- mrr                 命中名次的倒数，用于看"排得靠不靠前"
- relevance / is_refusal  只在启用生成时使用
"""
from __future__ import annotations

import re
from typing import List, Optional

REFUSAL_MARKERS = ["暂无法回答", "未检索到", "不足以回答", "无法回答"]

# 过于常见的字，参与匹配会把所有片段都判成命中
STOPWORDS = {"的", "是", "了", "在", "和", "与", "等", "为", "有", "由", "及", "以", "中"}

# 命中阈值：标准答案的关键字有这么多比例出现在同一片段里，就算这个片段命中了
HIT_RATIO = 0.4


def _tokenize(text: str) -> set:
    """中文按单字、英文按词。字符级重叠对同义改写不敏感，这是已知局限。"""
    if not text:
        return set()
    en = set(w.lower() for w in re.findall(r"[A-Za-z0-9]+", text))
    zh = set(re.findall(r"[\u4e00-\u9fa5]", text))
    return en | zh


def gold_tokens(gold_answer: Optional[str]) -> set:
    if not gold_answer:
        return set()
    return {t for t in _tokenize(gold_answer) if t not in STOPWORDS}


def chunk_hits(chunk_text: str, gold: set) -> bool:
    """单个片段是否算命中。"""
    if not gold:
        return False
    overlap = len(gold & _tokenize(chunk_text))
    return overlap >= max(1, len(gold) * HIT_RATIO)


def hit_rate(retrieved_chunks: List[dict], gold_answer: Optional[str]) -> bool:
    """Top-K 里是否有片段命中标准答案。没有标准答案时返回 False。"""
    gold = gold_tokens(gold_answer)
    return any(chunk_hits(c.get("text", ""), gold) for c in retrieved_chunks)


def first_hit_rank(retrieved_chunks: List[dict], gold_answer: Optional[str]) -> Optional[int]:
    """第一个命中片段的名次（1 起），没命中返回 None。"""
    gold = gold_tokens(gold_answer)
    for i, c in enumerate(retrieved_chunks, 1):
        if chunk_hits(c.get("text", ""), gold):
            return i
    return None


def keypoint_coverage(retrieved_chunks: List[dict], gold_answer: Optional[str]) -> float:
    """标准答案的去停用字字符有多少比例出现在 Top-K 里，取值 [0,1]。"""
    gold = gold_tokens(gold_answer)
    if not gold or not retrieved_chunks:
        return 0.0
    covered = set()
    for c in retrieved_chunks:
        covered |= gold & _tokenize(c.get("text", ""))
    return len(covered) / len(gold)


def relevance(answer: str, gold_answer: Optional[str]) -> float:
    """回答与标准答案的字符重叠度（归一化）。只在启用生成时有意义。"""
    if not answer:
        return 0.0
    if any(m in answer for m in REFUSAL_MARKERS):
        return 0.0
    if not gold_answer:
        return 0.5  # 无标准答案时给中性分，需人工复核
    gold = _tokenize(gold_answer)
    if not gold:
        return 0.5
    overlap = len(gold & _tokenize(answer))
    return round(min(1.0, overlap / max(1, len(gold) * 0.5)), 3)


def is_refusal(answer: str) -> bool:
    return any(m in answer for m in REFUSAL_MARKERS) if answer else True
