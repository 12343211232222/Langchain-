"""Retrieval 层统一接口。

上层（RAGService / 未来的 LangGraph 节点）只依赖 ``Retriever`` / ``Reranker``
抽象，不依赖具体 VectorStore 或 BM25 实现。
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from chatchat.rag.retrieval.schemas import RetrievalResult


class Retriever(ABC):
    """单路召回接口。"""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10, **kwargs) -> List[RetrievalResult]:
        """执行召回，返回统一 RetrievalResult 列表。"""
        raise NotImplementedError


class Reranker(ABC):
    """Rerank 接口（M05 仅预留，不实现具体模型）。"""

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """对融合结果重排序；``top_k`` 为 None 时不截断。"""
        raise NotImplementedError