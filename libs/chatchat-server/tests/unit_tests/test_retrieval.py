"""Hybrid Retrieval 统一层测试。

覆盖：Vector / BM25 / RRF / Weighted / Hybrid / NoOpReranker / Strategy / Regression
"""

from unittest.mock import MagicMock, patch

import pytest
from langchain.docstore.document import Document

from chatchat.core.exceptions import ServiceError, ValidationError
from chatchat.rag.retrieval import (
    BM25Retriever,
    FusionStrategy,
    HybridRetriever,
    NoOpReranker,
    RRFFusion,
    RetrievalResult,
    RetrievalStrategy,
    VectorRetriever,
    WeightedFusion,
    build_hybrid_retriever,
)
from chatchat.rag.retrieval.interfaces import Retriever, Reranker
from chatchat.rag.retrieval.schemas import RetrievalResult
from chatchat.rag import RAGQuery, CurrentRAGService, get_rag_service


class TestRetrievalInterface:
    def test_retriever_is_abstract(self):
        with pytest.raises(TypeError):
            Retriever()

    def test_reranker_is_abstract(self):
        with pytest.raises(TypeError):
            Reranker()


class _FakeKB:
    def __init__(self, docs):
        self.kb_name = "test_kb"
        self._docs = docs

    def search_docs(self, query, top_k, score_threshold):
        return self._docs

    def load_vector_store(self):
        inner_vs = MagicMock()
        inner_vs.docstore = MagicMock()
        inner_vs.docstore._dict = {str(i): doc for i, doc in enumerate(self._docs)}
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(return_value=inner_vs)
        ctx.__exit__ = MagicMock(return_value=False)
        holder = MagicMock()
        holder.acquire = MagicMock(return_value=ctx)
        return holder


class TestVectorRetriever:
    def test_vector_retrieve_returns_results(self):
        fake_docs = [
            Document(page_content="hello", metadata={"source": "a.txt", "score": 0.9}),
            Document(page_content="world", metadata={"source": "b.txt", "score": 0.8}),
        ]
        fake_kb = _FakeKB(fake_docs)
        with patch(
            "chatchat.rag.retrieval.vector.KBServiceFactory.get_service_by_name",
            return_value=fake_kb,
        ):
            retriever = VectorRetriever("kb")
            results = retriever.retrieve("q", top_k=3)
        assert len(results) == 2
        assert results[0].content == "hello"
        assert results[0].retriever == "vector"
        assert results[0].rank == 0
        assert results[1].rank == 1

    def test_vector_retrieve_unknown_kb(self):
        with patch(
            "chatchat.rag.retrieval.vector.KBServiceFactory.get_service_by_name",
            return_value=None,
        ):
            retriever = VectorRetriever("kb")
            assert retriever.retrieve("q", top_k=3) == []


class TestBM25Retriever:
    def test_bm25_keyword_match(self):
        docs = [
            Document(page_content="北京是中国的首都", metadata={"source": "cn.txt"}),
            Document(page_content="上海是金融中心", metadata={"source": "sh.txt"}),
            Document(page_content="Python 是一门编程语言", metadata={"source": "py.txt"}),
        ]
        retriever = BM25Retriever(docs)
        results = retriever.retrieve("北京 首都", top_k=2)
        assert len(results) >= 1
        for r in results:
            assert r.retriever == "bm25"
            assert isinstance(r.rank, int)

    def test_bm25_empty_documents(self):
        retriever = BM25Retriever([])
        assert retriever.retrieve("q", top_k=5) == []

    def test_bm25_no_result(self):
        docs = [Document(page_content="abcdef", metadata={"source": "x.txt"})]
        retriever = BM25Retriever(docs)
        results = retriever.retrieve("xyz", top_k=5)
        assert isinstance(results, list)  # 无匹配不抛异常


class TestRRFFusion:
    def test_rrf_overlapping_results(self):
        """Vector: A B C ; BM25: B C D → 重叠结果 B C 排名应合理。"""
        vector = [
            RetrievalResult(content="A", metadata={"s": "a"}, retriever="vector", rank=0),
            RetrievalResult(content="B", metadata={"s": "b"}, retriever="vector", rank=1),
            RetrievalResult(content="C", metadata={"s": "c"}, retriever="vector", rank=2),
        ]
        bm25 = [
            RetrievalResult(content="B", metadata={"s": "b"}, retriever="bm25", rank=0),
            RetrievalResult(content="C", metadata={"s": "c"}, retriever="bm25", rank=1),
            RetrievalResult(content="D", metadata={"s": "d"}, retriever="bm25", rank=2),
        ]
        fusion = RRFFusion(k=60)
        result = fusion.fuse([vector, bm25])
        contents = [r.content for r in result]
        assert "B" in contents
        assert "C" in contents
        # B 出现在两路召回中，排名应优于仅出现一路的 A/D
        assert contents.index("B") < contents.index("A")
        assert contents.index("B") < contents.index("D")
        for r in result:
            assert r.retriever == "fusion"
            assert r.score is not None
            assert isinstance(r.rank, int)

    def test_rrf_does_not_use_raw_score(self):
        """RRF 应基于 rank 而非 score。"""
        vector = [
            RetrievalResult(content="A", metadata={}, retriever="vector", rank=0, score=0.1),
        ]
        bm25 = [
            RetrievalResult(content="B", metadata={}, retriever="bm25", rank=0, score=100.0),
        ]
        result = RRFFusion(k=60).fuse([vector, bm25])
        # 两者都只在 rank=0 出现一次，RRF score 应相等
        assert result[0].score == result[1].score


class TestWeightedFusion:
    def test_weighted_fusion_weights_effective(self):
        """通过权重差异验证加权融合生效。"""
        vector = [
            RetrievalResult(content="A", metadata={}, retriever="vector", rank=0, score=0.8),
            RetrievalResult(content="B", metadata={}, retriever="vector", rank=1, score=0.6),
        ]
        bm25 = [
            RetrievalResult(content="A", metadata={}, retriever="bm25", rank=0, score=80.0),
            RetrievalResult(content="B", metadata={}, retriever="bm25", rank=1, score=60.0),
        ]
        # 让 vector 权重占绝对主导（0.9 vs 0.1），A 在 vector 排名更靠前，结果应也是 A 靠前
        fusion = WeightedFusion(weights=[0.9, 0.1])
        result = fusion.fuse([vector, bm25])
        assert result[0].content == "A"
        assert result[0].score is not None

    def test_weighted_normalization(self):
        """min-max 归一化：同一召回器内 score 差距应被压缩到 [0,1]。"""
        vector = [
            RetrievalResult(content="X", metadata={}, retriever="vector", rank=0, score=0.2),
            RetrievalResult(content="Y", metadata={}, retriever="vector", rank=1, score=0.1),
        ]
        fusion = WeightedFusion(weights=[1.0])
        result = fusion.fuse([vector])
        # 归一化后 X=1.0, Y=0.0（因为 max=0.2, min=0.1）
        assert result[0].content == "X"
        assert result[0].score == pytest.approx(1.0, rel=1e-3)

    def test_weighted_invalid_weights(self):
        with pytest.raises(ValidationError):
            WeightedFusion(weights=[0.7, -0.1])
        with pytest.raises(ValidationError):
            WeightedFusion(weights=[0.0, 0.0])
        with pytest.raises(ValidationError):
            WeightedFusion(weights=[0.3, 0.5]).fuse([[], [], []])  # 路数与权重不匹配


class TestNoOpReranker:
    def test_noop_does_not_change_content(self):
        results = [
            RetrievalResult(content="a", metadata={}, score=0.5),
            RetrievalResult(content="b", metadata={}, score=0.3),
        ]
        out = NoOpReranker().rerank("q", results, top_k=None)
        assert [r.content for r in out] == ["a", "b"]

    def test_noop_topk_truncation(self):
        results = [RetrievalResult(content=str(i), metadata={}, score=i) for i in range(10)]
        out = NoOpReranker().rerank("q", results, top_k=3)
        assert len(out) == 3


class TestHybridRetriever:
    def test_hybrid_vector_and_bm25(self):
        docs = [
            Document(page_content="人工智能", metadata={"source": "ai.txt"}),
            Document(page_content="深度学习", metadata={"source": "dl.txt"}),
        ]
        vector_retriever = BM25Retriever(docs)  # 用 BM25 模拟 vector，确保都能工作
        bm25_retriever = BM25Retriever(docs)
        hybrid = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            fusion_strategy=RRFFusion(k=60),
            reranker=NoOpReranker(),
        )
        results = hybrid.retrieve("人工智能", top_k=5)
        assert len(results) >= 1
        for r in results:
            assert r.retriever == "fusion"


class TestRetrievalStrategy:
    def test_strategy_vector(self):
        assert RetrievalStrategy.VECTOR.value == "vector"

    def test_strategy_hybrid_rrf(self):
        assert RetrievalStrategy.HYBRID_RRF.value == "hybrid_rrf"

    def test_strategy_hybrid_weighted(self):
        assert RetrievalStrategy.HYBRID_WEIGHTED.value == "hybrid_weighted"


class TestRAGServiceStrategy:
    def test_default_vector_strategy(self):
        """默认策略（None）仍走原有向量路径，行为不变。"""
        fake_docs = [Document(page_content="hello", metadata={"source": "a.txt", "score": 0.9})]
        fake_kb = _FakeKB(fake_docs)
        with patch(
            "chatchat.rag.service.KBServiceFactory.get_service_by_name",
            return_value=fake_kb,
        ):
            resp = CurrentRAGService().retrieve(
                RAGQuery(query="q", knowledge_base="kb", top_k=3)
            )
        assert len(resp.documents) == 1
        assert resp.metadata.get("retrieval_strategy") == "vector"

    def test_unknown_strategy_raises(self):
        with patch(
            "chatchat.rag.service.KBServiceFactory.get_service_by_name",
            return_value=_FakeKB([]),
        ):
            with pytest.raises(ValidationError, match="未知检索策略"):
                CurrentRAGService().retrieve(
                    RAGQuery(query="q", knowledge_base="kb", retrieval_strategy="unknown")
                )