"""RAG 统一接口层测试：Schema / Interface / Adapter / Empty / Exception。"""

from unittest.mock import patch

import pytest
from langchain.docstore.document import Document

from chatchat.rag import (
    CurrentRAGService,
    RAGQuery,
    RAGDocument,
    RAGResponse,
    RAGServiceInterface,
    get_rag_service,
)


class TestRAGSchema:
    def test_rag_query_create(self):
        q = RAGQuery(query="hello", knowledge_base="kb", top_k=5)
        assert q.query == "hello"
        assert q.knowledge_base == "kb"
        assert q.top_k == 5
        assert q.score_threshold is None

    def test_rag_document_create(self):
        d = RAGDocument(content="text", score=0.8, metadata={"source": "a.txt"})
        assert d.content == "text"
        assert d.score == 0.8
        assert d.metadata["source"] == "a.txt"

    def test_rag_response_create(self):
        resp = RAGResponse(query="hello", documents=[])
        assert resp.query == "hello"
        assert resp.documents == []


class TestRAGInterface:
    def test_current_rag_service_implements_interface(self):
        assert issubclass(CurrentRAGService, RAGServiceInterface)

    def test_get_rag_service_returns_interface(self):
        svc = get_rag_service()
        assert isinstance(svc, RAGServiceInterface)


class _FakeKB:
    def __init__(self, docs):
        self._docs = docs

    def search_docs(self, query, top_k, score_threshold):
        return self._docs


class TestRAGAdapter:
    def _patch_factory(self, fake_kb):
        return patch(
            "chatchat.rag.service.KBServiceFactory.get_service_by_name",
            return_value=fake_kb,
        )

    def test_retrieve_converts_documents(self):
        fake_kb = _FakeKB(
            [Document(page_content="hello world", metadata={"source": "a.txt", "score": 0.9})]
        )
        with self._patch_factory(fake_kb):
            resp = CurrentRAGService().retrieve(
                RAGQuery(query="q", knowledge_base="kb", top_k=3, score_threshold=0.5)
            )
        assert isinstance(resp, RAGResponse)
        assert len(resp.documents) == 1
        assert resp.documents[0].content == "hello world"
        assert resp.documents[0].score == 0.9
        assert resp.documents[0].metadata["source"] == "a.txt"

    def test_retrieve_empty_result(self):
        fake_kb = _FakeKB([])
        with self._patch_factory(fake_kb):
            resp = CurrentRAGService().retrieve(
                RAGQuery(query="q", knowledge_base="kb")
            )
        assert resp.documents == []

    def test_retrieve_does_not_swallow_exception(self):
        class _BoomKB:
            def search_docs(self, query, top_k, score_threshold):
                raise ValueError("boom")

        with self._patch_factory(_BoomKB()):
            with pytest.raises(ValueError, match="boom"):
                CurrentRAGService().retrieve(RAGQuery(query="q", knowledge_base="kb"))