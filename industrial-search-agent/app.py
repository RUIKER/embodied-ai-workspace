"""
Gradio Web demo 入口。

提供:
- 问答输入框 → 显示回答 + 检索来源 + 延迟
- 检索模式切换(bm25 / vector / hybrid / rerank),可直观对比四档效果
- 展示检索到的原文片段

运行:python app.py
"""
from __future__ import annotations
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr
from rag.chain import answer


def _format_sources(sources):
    lines = []
    for i, s in enumerate(sources, 1):
        lines.append(f"[{i}] 来源:{s.get('source','未知')} | 分类:{s.get('category','未知')} | 分数:{s.get('score','—')}")
    return "\n".join(lines) if lines else "无"


def _format_chunks(retrieved):
    lines = []
    for i, r in enumerate(retrieved, 1):
        lines.append(f"[{i}] ({r.get('category','')}) {r['text']}")
    return "\n\n".join(lines) if lines else "无"


def chat(question, mode):
    if not question.strip():
        return "请输入问题。", "", ""
    res = answer(question, mode=mode)
    meta = f"检索模式:{res['mode']}  |  延迟:{res['latency_ms']} ms  |  召回:{len(res['retrieved'])} 条"
    return res["answer"], _format_sources(res["sources"]), meta + "\n\n" + _format_chunks(res["retrieved"])


def main():
    with gr.Blocks(title="工业知识库 RAG Agent") as demo:
        gr.Markdown("# 工业具身智能知识库 RAG 问答系统\n基于 hybrid 检索 + rerank + GLM 生成。")
        with gr.Row():
            inp = gr.Textbox(label="问题", placeholder="例:谐波减速器目前被哪个国家垄断?", lines=2)
        with gr.Row():
            mode = gr.Radio(["bm25", "vector", "hybrid", "rerank"], value="rerank", label="检索模式")
            btn = gr.Button("提问", variant="primary")
        out_answer = gr.Textbox(label="回答", lines=8)
        with gr.Row():
            out_sources = gr.Textbox(label="引用来源", lines=6)
            out_chunks = gr.Textbox(label="检索片段", lines=6)
        gr.Markdown(
            f"> 模式说明:**bm25**=关键词(原系统基线) · **vector**=纯向量 · **hybrid**=向量+BM25融合 · **rerank**=hybrid+重排(最终方案)\n\n"
            f"> 生成:{'智谱 GLM API' if os.environ.get('ZHIPUAI_API_KEY') else '降级模式(未配置 ZHIPUAI_API_KEY,返回检索片段)'}"
        )
        btn.click(chat, [inp, mode], [out_answer, out_sources, out_chunks])
        inp.submit(chat, [inp, mode], [out_answer, out_sources, out_chunks])
    demo.launch()


if __name__ == "__main__":
    main()
