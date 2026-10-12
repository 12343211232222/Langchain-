"""向量召回适配器。

复用现有知识库向量检索（``KBService.search_docs``），将结果适配到统一
``Retriever`` 接口。不修改底层 VectorStore / Embedding，不改变默认
top_k 与 score 行为。
"""

from typing import List, Optional

from chatchat.settings import Settings
from chatchat.rag.retrieval.interfaces import Retriever
from chatchat.rag.retrieval.schemas import RetrievalResult
from chatchat.server.knowledge_base.kb_service.base import KBServiceFactory


class VectorRetriever(Retriever):
    """基于现有 KBService 向量检索的召回适配器（Adapter）。"""

    def __init__(self, kb_name: str):
        self.kb_name = kb_name

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        score_threshold: Optional[float] = None,
        **kwargs,
    ) -> List[RetrievalResult]:
        kb = KBServiceFactory.get_service_by_name(self.kb_name)
        if kb is None:
            return []
        if score_threshold is None:
            score_threshold = Settings.kb_settings.SCORE_THRESHOLD
        raw_docs = kb.search_docs(query, top_k, score_threshold)
        return [
            RetrievalResult(
                content=doc.page_content,
                score=doc.metadata.get("score"),
                metadata=dict(doc.metadata),
                retriever="vector",
                rank=i,
            )
            for i, doc in enumerate(raw_docs)
        ]