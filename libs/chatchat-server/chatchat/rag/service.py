"""当前 RAG 能力的 Adapter 实现。

复用现有知识库检索（``KBService.search_docs``），将结果统一转换为
``RAGResponse``。本类不复制检索逻辑、不做 LLM 生成、不吞异常。
"""

import time

from chatchat.settings import Settings
from chatchat.core.exceptions import ValidationError
from chatchat.rag.interfaces import RAGServiceInterface
from chatchat.rag.schemas import RAGDocument, RAGQuery, RAGResponse
from chatchat.rag.retrieval.strategy import RetrievalStrategy, build_hybrid_retriever
from chatchat.server.knowledge_base.kb_service.base import KBServiceFactory


class CurrentRAGService(RAGServiceInterface):
    """基于现有知识库检索的 RAG Service 适配器（Adapter）。

    默认（``retrieval_strategy=None`` 或 ``"vector"``）仍走原有向量检索，
    行为不变；显式指定 ``"hybrid_rrf"`` / ``"hybrid_weighted"`` 时才走
    Vector + BM25 多路召回与融合。
    """

    def retrieve(self, request: RAGQuery) -> RAGResponse:
        start = time.perf_counter()

        strategy = request.retrieval_strategy or RetrievalStrategy.VECTOR.value

        documents: list[RAGDocument] = []
        if request.knowledge_base:
            kb = KBServiceFactory.get_service_by_name(request.knowledge_base)
            if kb is not None:
                score_threshold = (
                    request.score_threshold
                    if request.score_threshold is not None
                    else Settings.kb_settings.SCORE_THRESHOLD
                )
                if strategy == RetrievalStrategy.VECTOR.value:
                    raw_docs = kb.search_docs(
                        request.query,
                        request.top_k,
                        score_threshold,
                    )
                    documents = [
                        RAGDocument(
                            content=doc.page_content,
                            score=doc.metadata.get("score"),
                            metadata=dict(doc.metadata),
                        )
                        for doc in raw_docs
                    ]
                elif strategy in (
                    RetrievalStrategy.HYBRID_RRF.value,
                    RetrievalStrategy.HYBRID_WEIGHTED.value,
                ):
                    hybrid = build_hybrid_retriever(kb, strategy=strategy)
                    results = hybrid.retrieve(
                        request.query,
                        top_k=request.top_k,
                        score_threshold=score_threshold,
                    )
                    documents = [
                        RAGDocument(
                            content=r.content,
                            score=r.score,
                            metadata=dict(r.metadata),
                        )
                        for r in results
                    ]
                else:
                    raise ValidationError(f"未知检索策略: {strategy}")

        elapsed = time.perf_counter() - start
        return RAGResponse(
            query=request.query,
            documents=documents,
            retrieval_time=elapsed,
            metadata={
                "provider": "current",
                "knowledge_base": request.knowledge_base,
                "retrieval_strategy": strategy,
            },
        )