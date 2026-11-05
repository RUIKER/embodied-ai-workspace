"""
项目集中配置:路径、模型、切分参数、检索参数。
所有模块统一从此处读取配置,避免散落硬编码。
"""
import os
from pathlib import Path

# 国内 HuggingFace 镜像,加速模型下载(bge embedding / reranker)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# sentence-transformers 走 HF 镜像
os.environ.setdefault("HF_HUB_ENDPOINT", "https://hf-mirror.com")

# ===== 路径配置 =====
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# 各目录支持用环境变量覆盖，便于冒烟测试把产物写到 data/smoke/，
# 不碰真实语料的 processed / chroma_db。
RAW_DIR = Path(os.environ.get("KB_RAW_DIR", DATA_DIR / "raw"))              # 原始知识库文档(用户放入)
PROCESSED_DIR = Path(os.environ.get("KB_PROCESSED_DIR", DATA_DIR / "processed"))  # 清洗切分后的 chunk
QUESTIONS_DIR = DATA_DIR / "questions"     # 100 个测试问题
CHROMA_DIR = Path(os.environ.get("KB_CHROMA_DIR", DATA_DIR / "chroma_db"))  # Chroma 持久化索引
CHROMA_COLLECTION = os.environ.get("KB_COLLECTION", "industrial_knowledge")

CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
QUESTIONS_PATH = Path(os.environ.get("KB_QUESTIONS", QUESTIONS_DIR / "test_questions.json"))

# ===== 支持的原始文档格式 =====
SUPPORTED_EXTS = {".txt", ".md", ".docx", ".pdf"}

# ===== chunk 切分参数 =====
CHUNK_SIZE = 600        # 500~800 字区间,取 600
CHUNK_OVERLAP = 80      # 重叠,保证上下文连续性

# ===== 模型配置(第2周使用) =====
EMBED_MODEL = "BAAI/bge-small-zh-v1.5"     # 中文 embedding
RERANK_MODEL = "BAAI/bge-reranker-base"    # 中文 cross-encoder 重排序

# ===== 检索参数(第2周使用) =====
TOP_K_VECTOR = 20       # 向量检索召回数
TOP_K_BM25 = 20         # 关键词检索召回数
FINAL_TOP_K = 5         # rerank 后最终返回数
VECTOR_WEIGHT = 0.6     # hybrid 融合时向量权重
BM25_WEIGHT = 0.4       # hybrid 融合时关键词权重

# ===== LLM 生成配置(第2周使用) =====
# 智谱 GLM API,从环境变量读取 Key
GLM_MODEL = "glm-4.6"   # 生成模型
# os.environ["ZHIPUAI_API_KEY"] 需用户自行设置


def ensure_dirs():
    """确保所有数据目录存在。"""
    for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, QUESTIONS_DIR, CHROMA_DIR]:
        d.mkdir(parents=True, exist_ok=True)
