"""BM25 召回适配器。

项目现有 BM25 实现位于 ``chatchat.server.file_rag.retrievers.ensemble``
（langchain_community BM25Retriever + jieba 分词 + rank_bm25）。
本类复用同一套实现，不重新实现 BM25，不引入 Elasticsearch / OpenSearch。

BM25 只负责召回，不负责 Embedding / Rerank / LLM / Prompt。
"""

from typing import Callable, List, Optional

from langchain.docstore.document import Document

from chatchat.rag.retrieval.interfaces import Retriever
from chatchat.rag.retrieval.schemas import RetrievalResult


class BM25Retriever(Retriever):
    """基于现有 langchain_community BM25Retriever 的召回适配器（Adapter）。

    :param documents: 语料文档列表（由调用方提供，本类不访问 VectorStore）。
    :param preprocess_func: 分词函数，默认 ``jieba.lcut_for_search``（与项目现有实现一致）。
    """

    def __init__(
        self,
        documents: List[Document],
        preprocess_func: Optional[Callable] = None,
    ):
        self._documents = list(documents)
        self._retriever = None
        if preprocess_func is None:
            import jieba

            preprocess_func = jieba.lcut_for_search
        self._preprocess = preprocess_func
        if self._documents:
            from langchain_community.retrievers import BM25Retriever as _LCBM25Retriever

            self._retriever = _LCBM25Retriever.from_documents(
                self._documents, preprocess_func=self._preprocess
            )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        **kwargs,
    ) -> List[RetrievalResult]:
        if self._retriever is None:
            return []
        self._retriever.k = top_k
        docs = self._retriever.invoke(query)[:top_k]

        # 通过底层 rank_bm25 vectorizer 取回原始 BM25 分数（供 Weighted Fusion 使用），
        # 语料与返回文档为同一批对象，可按对象 id 对齐。
        score_by_id = {}
        try:
            scores = self._retriever.vectorizer.get_scores(self._preprocess(query))
            score_by_id = {
                id(doc): float(scores[i]) for i, doc in enumerate(self._documents)
            }
        except Exception:
            score_by_id = {}

        return [
            RetrievalResult(
                content=doc.page_content,
                score=score_by_id.get(id(doc)),
                metadata=dict(doc.metadata),
                retriever="bm25",
                rank=i,
            )
            for i, doc in enumerate(docs)
        ]