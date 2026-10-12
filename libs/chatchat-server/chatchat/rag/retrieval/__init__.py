"""Hybrid Retrieval 统一层。

M05 提供：
- ``schemas``：RetrievalResult（扩展 M02 RAGDocument）
- ``interfaces``：Retriever / Reranker 抽象接口
- ``vector``：VectorRetriever（复用 KBService 向量检索）
- ``bm25``：BM25Retriever（复用现有 langchain_community BM25 实现）
- ``fusion``：FusionStrategy / RRFFusion / WeightedFusion
- ``rerank``：NoOpReranker（Rerank Hook）
- ``strategy``：RetrievalStrategy / HybridRetriever / build_hybrid_retriever
"""

from chatchat.rag.retrieval.schemas import RetrievalResult
from chatchat.rag.retrieval.interfaces import Retriever, Reranker
from chatchat.rag.retrieval.vector import VectorRetriever
from chatchat.rag.retrieval.bm25 import BM25Retriever
from chatchat.rag.retrieval.fusion import FusionStrategy, RRFFusion, WeightedFusion
from chatchat.rag.retrieval.rerank import NoOpReranker
from chatchat.rag.retrieval.strategy import (
    RetrievalStrategy,
    HybridRetriever,
    build_hybrid_retriever,
)

__all__ = [
    "RetrievalResult",
    "Retriever",
    "Reranker",
    "VectorRetriever",
    "BM25Retriever",
    "FusionStrategy",
    "RRFFusion",
    "WeightedFusion",
    "NoOpReranker",
    "RetrievalStrategy",
    "HybridRetriever",
    "build_hybrid_retriever",
]