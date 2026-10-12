"""检索策略与 HybridRetriever。

策略枚举：
- ``VECTOR``：现有向量检索（默认，行为不变）
- ``HYBRID_RRF``：Vector + BM25 → RRF 融合
- ``HYBRID_WEIGHTED``：Vector + BM25 → Weighted 融合

HybridRetriever 只负责多路召回与融合，不负责 Chunk / Embedding /
VectorStore 创建 / Document Parsing / Rerank 模型加载 / LLM。
"""

from enum import Enum
from typing import List, Optional

from chatchat.core.exceptions import ServiceError
from chatchat.rag.retrieval.bm25 import BM25Retriever
from chatchat.rag.retrieval.fusion import FusionStrategy, RRFFusion, WeightedFusion
from chatchat.rag.retrieval.interfaces import Retriever, Reranker
from chatchat.rag.retrieval.rerank import NoOpReranker
from chatchat.rag.retrieval.schemas import RetrievalResult
from chatchat.rag.retrieval.vector import VectorRetriever


class RetrievalStrategy(str, Enum):
    """检索策略标识。"""

    VECTOR = "vector"
    HYBRID_RRF = "hybrid_rrf"
    HYBRID_WEIGHTED = "hybrid_weighted"


class HybridRetriever(Retriever):
    """多路召回 + 融合的复合 Retriever。"""

    def __init__(
        self,
        vector_retriever: Retriever,
        bm25_retriever: Retriever,
        fusion_strategy: FusionStrategy,
        reranker: Optional[Reranker] = None,
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.fusion_strategy = fusion_strategy
        self.reranker = reranker or NoOpReranker()

    def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[RetrievalResult]:
        results = [
            self.vector_retriever.retrieve(query, top_k=top_k, **kwargs),
            self.bm25_retriever.retrieve(query, top_k=top_k, **kwargs),
        ]
        fused = self.fusion_strategy.fuse(results)
        return self.reranker.rerank(query, fused, top_k=top_k)


def _get_kb_corpus(kb) -> list:
    """获取知识库语料（供 BM25 使用）。

    复用项目现有做法（``ensemble.py`` 从 FAISS docstore 取全部文档）。
    仅支持暴露 docstore 的向量库；其他向量库显式报错，不做静默回退。
    """
    load_vs = getattr(kb, "load_vector_store", None)
    if load_vs is not None:
        with load_vs().acquire() as vs:
            docstore = getattr(vs, "docstore", None)
            if docstore is not None and hasattr(docstore, "_dict"):
                return list(docstore._dict.values())
    raise ServiceError(
        f"知识库 {getattr(kb, 'kb_name', '?')} 的向量库不支持导出语料，"
        "暂无法使用 hybrid 检索策略"
    )


def build_hybrid_retriever(
    kb,
    strategy: str,
    rrf_k: int = 60,
    weights: Optional[List[float]] = None,
) -> HybridRetriever:
    """按策略构建知识库的 HybridRetriever（Vector + BM25 + Fusion + NoOpReranker）。"""
    vector_retriever = VectorRetriever(kb.kb_name)
    bm25_retriever = BM25Retriever(_get_kb_corpus(kb))
    if strategy == RetrievalStrategy.HYBRID_RRF.value:
        fusion: FusionStrategy = RRFFusion(k=rrf_k)
    elif strategy == RetrievalStrategy.HYBRID_WEIGHTED.value:
        fusion = WeightedFusion(weights=weights or [0.7, 0.3])
    else:
        raise ServiceError(f"非 hybrid 检索策略: {strategy}")
    return HybridRetriever(
        vector_retriever=vector_retriever,
        bm25_retriever=bm25_retriever,
        fusion_strategy=fusion,
    )