"""Rerank Hook。

M05 仅提供接口与 NoOp 默认实现，不加载 BGE / Cross Encoder / RAGFlow Reranker。
"""

from typing import List, Optional

from chatchat.rag.retrieval.interfaces import Reranker
from chatchat.rag.retrieval.schemas import RetrievalResult


class NoOpReranker(Reranker):
    """默认 Reranker：不改变结果内容与顺序，仅按 top_k 截断。"""

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        if top_k is None:
            return list(results)
        return list(results)[:top_k]