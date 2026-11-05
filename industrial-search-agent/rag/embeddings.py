"""
共享的 embedding / reranker 模型加载,保证 build_index 与 retriever 用同一模型。

模型来源:优先用 modelscope 本地缓存(国内稳定,绕开 huggingface_hub 元数据校验问题);
若本地已有缓存则直接用,否则触发 modelscope 下载。build_index 与 retriever 共用同一模型。
"""
from __future__ import annotations
import functools
from config import EMBED_MODEL, RERANK_MODEL


def _local_model_path(repo_id: str) -> str:
    """用 modelscope 获取模型本地路径(已缓存则直接返回,否则下载)。"""
    from modelscope import snapshot_download
    return snapshot_download(repo_id)


@functools.lru_cache(maxsize=1)
def get_embedding_function():
    """Chroma 用的 SentenceTransformer embedding function(单例)。"""
    from chromadb.utils import embedding_functions
    local_path = _local_model_path(EMBED_MODEL)
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=local_path)


@functools.lru_cache(maxsize=1)
def get_reranker():
    """bge-reranker cross-encoder(单例),用于 rerank。"""
    from sentence_transformers import CrossEncoder
    local_path = _local_model_path(RERANK_MODEL)
    return CrossEncoder(local_path)


def embed_query(text: str):
    """单条查询向量化。"""
    ef = get_embedding_function()
    return ef([text])[0]
