"""
第2周:LLM 生成回答。

默认调用智谱 GLM API(glm-4.6),读取环境变量 ZHIPUAI_API_KEY。
按检索到的上下文 + 用户问题,生成结构化回答并附来源。

无 API Key 时降级:直接返回检索到的 top1 片段(保证链路可演示)。
"""
from __future__ import annotations
import os
from typing import List, Optional
from config import GLM_MODEL

# 系统 prompt:约束模型基于检索上下文回答,给出来源,避免幻觉
SYSTEM_PROMPT = """你是工业具身智能机器人领域的知识问答助手。请严格基于【参考资料】回答用户问题。

要求:
1. 只能使用参考资料中的信息,不得编造。
2. 回答简洁准确,分点表述如果合适。
3. 如果参考资料不足以回答,明确说"根据现有知识库暂无法回答",不要臆测。
4. 回答末尾用【来源】标注引用的资料编号。
"""

USER_PROMPT_TEMPLATE = """【参考资料】
{context}

【用户问题】
{question}
"""


def _build_context(contexts: List[dict]) -> str:
    """把检索片段拼成带编号的上下文。"""
    lines = []
    for i, c in enumerate(contexts, 1):
        lines.append(f"[{i}] (来源:{c.get('source','未知')} | 分类:{c.get('category','未知')})\n{c['text']}")
    return "\n\n".join(lines)


def build_messages(question: str, contexts: List[dict]) -> list:
    """构造 GLM chat messages。"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT_TEMPLATE.format(
            context=_build_context(contexts), question=question)},
    ]


def _call_glm(messages: list) -> Optional[str]:
    """调用智谱 GLM API,失败返回 None。"""
    api_key = os.environ.get("ZHIPUAI_API_KEY")
    if not api_key:
        return None
    try:
        from zhipuai import ZhipuAI
        client = ZhipuAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=GLM_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=512,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"[generator] GLM API 调用失败: {e}")
        return None


def _fallback_answer(question: str, contexts: List[dict]) -> str:
    """无 API Key 或调用失败时的降级:返回 top1 片段拼接。"""
    if not contexts:
        return "根据现有知识库暂无法回答(未检索到相关资料)。"
    top = contexts[0]
    return (
        f"(降级模式:未配置 ZHIPUAI_API_KEY,以下为检索到的最相关资料)\n\n"
        f"{top['text']}\n\n【来源】{top.get('source','未知')} | {top.get('category','未知')}"
    )


def generate(question: str, contexts: List[dict]) -> str:
    """根据问题与检索上下文生成回答。"""
    if not contexts:
        return "根据现有知识库暂无法回答(未检索到相关资料)。"
    messages = build_messages(question, contexts)
    answer = _call_glm(messages)
    if answer:
        return answer
    return _fallback_answer(question, contexts)


if __name__ == "__main__":
    # demo:用假上下文测试 prompt 构造与降级
    fake = [{"text": "谐波减速器是工业机器人核心零部件,目前主要被日本企业垄断。", "source": "test.docx", "category": "产业链"}]
    print("=== 降级模式输出 ===")
    print(generate("谐波减速器被谁垄断?", fake))
