"""
第1周核心:文档加载 → 清洗 → chunk 切分 → 打标签 → 写 jsonl。

输入:data/raw/ 下的 .txt/.md/.docx/.pdf
输出:data/processed/chunks.jsonl,每行:
    {"id": "...", "text": "...", "category": "...", "layer": "...", "source": "文件名"}

设计要点:
- 多格式加载器,统一返回纯文本。
- 清洗:合并多余空白、去空行,保留语义。
- 切分:RecursiveCharacterTextSplitter,中文友好 separators。
- 打标签:schema.classify_tag,每个 chunk 独立打标。
- 幂等:每次重跑覆盖输出文件;chunk id 用 source+序号哈希稳定生成。
"""
from __future__ import annotations
import json
import re
import hashlib
from pathlib import Path
from typing import List

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))

from config import RAW_DIR, PROCESSED_DIR, CHUNKS_PATH, SUPPORTED_EXTS, CHUNK_SIZE, CHUNK_OVERLAP, ensure_dirs
from knowledge.schema import classify_tag

# 中文友好的切分分隔符:优先按段落/句号切,尽量不切断语义
SEPARATORS = ["\n\n", "\n", "。", "!", "?", ";", ";", "，", " ", ""]


# ===== 1. 多格式文档加载 =====
def load_text(path: Path) -> str:
    """根据扩展名加载文档,返回纯文本。"""
    ext = path.suffix.lower()
    if ext in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".docx":
        return _load_docx(path)
    if ext == ".pdf":
        return _load_pdf(path)
    raise ValueError(f"不支持的文件格式: {ext}")


def _load_docx(path: Path) -> str:
    import docx
    doc = docx.Document(str(path))
    parts = [p.text for p in doc.paragraphs if p.text.strip()]
    # 表格内容也提取
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n".join(parts)


def _load_pdf(path: Path) -> str:
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    parts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            parts.append(t)
    return "\n".join(parts)


# ===== 2. 清洗 =====
def clean_text(text: str) -> str:
    """去除多余空白、连续空行、不可见字符,保留语义。"""
    # 去除常见不可见字符
    text = text.replace("　", " ")  # 全角空格
    # 合并连续空白为单个空格(但保留换行)
    text = re.sub(r"[ \t]+", " ", text)
    # 连续空行压缩为单个换行
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 去除行首尾空白
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(ln for ln in lines if ln)
    return text.strip()


# ===== 3. chunk 切分 =====
def split_text(text: str) -> List[str]:
    """用 RecursiveCharacterTextSplitter 切分,中文友好。"""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        # 第1周若未装 langchain,退化为简单按长度切分
        return _fallback_split(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=SEPARATORS,
        keep_separator=True,
    )
    chunks = splitter.split_text(text)
    # 过滤过短的碎片(<30 字基本无信息量)
    return [c.strip() for c in chunks if len(c.strip()) >= 30]


def _fallback_split(text: str) -> List[str]:
    """无 langchain 时的退化切分:先按段落累积,超长段落按长度滚动切(带 overlap)。"""
    # 优先按换行切成段落
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: List[str] = []
    buf = ""
    for p in paragraphs:
        # 单段本身超长 → 先把 buf 冲掉,再对该段按长度滚动切
        if len(p) > CHUNK_SIZE:
            if buf:
                chunks.append(buf)
                buf = ""
            chunks.extend(_rolling_split(p))
            continue
        if len(buf) + len(p) + 1 <= CHUNK_SIZE:
            buf = (buf + "\n" + p).strip() if buf else p
        else:
            if buf:
                chunks.append(buf)
            buf = p
    if buf:
        chunks.append(buf)
    return [c for c in chunks if len(c) >= 30]


def _rolling_split(text: str) -> List[str]:
    """对超长文本按 CHUNK_SIZE 滚动切,带 CHUNK_OVERLAP 重叠,尽量在句号处断开。"""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + CHUNK_SIZE, n)
        # 尽量在句号/换行处断开,避免切断句子
        if end < n:
            for sep in ["。", "！", "？", "\n", "；", "，"]:
                pos = text.rfind(sep, start, end)
                if pos > start + CHUNK_SIZE // 2:
                    end = pos + 1
                    break
        chunks.append(text[start:end].strip())
        if end >= n:
            break
        start = end - CHUNK_OVERLAP  # overlap
    return [c for c in chunks if len(c) >= 30]


# ===== 4. chunk id 稳定生成 =====
def make_chunk_id(source: str, idx: int, text: str) -> str:
    raw = f"{source}::{idx}::{text[:32]}"
    h = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    return f"{Path(source).stem}_{idx:04d}_{h}"


# ===== 5. 主流程 =====
def ingest() -> dict:
    """遍历 data/raw/ 处理所有文档,输出 chunks.jsonl,返回统计信息。"""
    ensure_dirs()
    if not CHUNKS_PATH.parent.exists():
        CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)

    files = [p for p in sorted(RAW_DIR.rglob("*")) if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS]
    if not files:
        print(f"[警告] {RAW_DIR} 下没有可处理的文档(支持 {sorted(SUPPORTED_EXTS)})。")
        print("       请把原始知识库文档放入 data/raw/ 后重跑。")
        return {"total": 0, "by_category": {}, "by_source": {}}

    all_chunks = []
    for fp in files:
        try:
            raw = load_text(fp)
        except Exception as e:
            print(f"[跳过] 读取失败 {fp.name}: {e}")
            continue
        text = clean_text(raw)
        if not text:
            continue
        pieces = split_text(text)
        source = fp.name
        for i, piece in enumerate(pieces):
            tags = classify_tag(piece)
            all_chunks.append({
                "id": make_chunk_id(source, i, piece),
                "text": piece,
                "category": tags["category"],
                "layer": tags["layer"],
                "source": source,
            })
        print(f"[完成] {source}: {len(pieces)} 个 chunk")

    # 写 jsonl
    with open(CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # 统计
    from collections import Counter
    by_category = Counter(c["category"] for c in all_chunks)
    by_source = Counter(c["source"] for c in all_chunks)

    print("\n========== 切分统计 ==========")
    print(f"总 chunk 数: {len(all_chunks)}")
    print("\n按八大类分布:")
    for k, v in by_category.most_common():
        print(f"  {k}: {v}")
    print("\n按来源文档分布:")
    for k, v in by_source.most_common():
        print(f"  {k}: {v}")
    print(f"\n输出文件: {CHUNKS_PATH}")

    return {"total": len(all_chunks), "by_category": dict(by_category), "by_source": dict(by_source)}


if __name__ == "__main__":
    ingest()
