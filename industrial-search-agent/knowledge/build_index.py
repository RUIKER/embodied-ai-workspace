"""
第2周:构建 Chroma 向量索引。

读取 data/processed/chunks.jsonl,用 BAAI/bge-small-zh-v1.5 生成 embedding,
持久化写入 data/chroma_db/(collection: industrial_knowledge)。

metadata 含 id/category/layer/source,供第2周检索过滤、第4周分维度评估。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import CHUNKS_PATH, CHROMA_DIR, CHROMA_COLLECTION, ensure_dirs
from rag.embeddings import get_embedding_function


def load_chunks(path: Path = CHUNKS_PATH):
    """从 jsonl 加载所有 chunk。"""
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path},请先运行 knowledge/ingest.py")
    chunks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    return chunks


def build_index():
    """构建并持久化 Chroma 索引。"""
    ensure_dirs()
    import chromadb

    chunks = load_chunks()
    print(f"加载 {len(chunks)} 个 chunk")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    ef = get_embedding_function()

    # 重建:先删旧 collection 再建,保证幂等
    try:
        client.delete_collection(CHROMA_COLLECTION)
    except Exception:
        pass
    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=ef,
        metadata={"description": "工业具身智能知识库"},
    )

    # Chroma 批量 upsert(分批避免单次过大)
    batch = 200
    for i in range(0, len(chunks), batch):
        batch_chunks = chunks[i:i + batch]
        collection.upsert(
            ids=[c["id"] for c in batch_chunks],
            documents=[c["text"] for c in batch_chunks],
            metadatas=[{
                "category": c["category"],
                "layer": c["layer"],
                "source": c["source"],
            } for c in batch_chunks],
        )
        print(f"  已写入 {min(i + batch, len(chunks))}/{len(chunks)}")

    print(f"\n索引构建完成: {CHROMA_DIR}")
    print(f"collection: {CHROMA_COLLECTION}, 共 {collection.count()} 条")
    return collection


if __name__ == "__main__":
    build_index()
