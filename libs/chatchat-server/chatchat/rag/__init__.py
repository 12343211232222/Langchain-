"""ChatChat RAG 统一接口层。

将现有知识库 / File RAG / 检索能力抽象为统一 RAG Service，
为后续接入 RAGFlow 文档解析、Hybrid Retrieval、Rerank、Citation 做准备。

M02 仅提供：
- ``schemas``：RAGQuery / RAGDocument / RAGResponse
- ``interfaces``：RAGServiceInterface
- ``service``：CurrentRAGService（复用现有检索的 Adapter）
- ``provider``：get_rag_service
"""

from chatchat.rag.schemas import RAGQuery, RAGDocument, RAGResponse
from chatchat.rag.interfaces import RAGServiceInterface
from chatchat.rag.service import CurrentRAGService
from chatchat.rag.provider import get_rag_service
from chatchat.rag.parser import DocumentParser, CurrentDocumentParser
from chatchat.rag.chunk import Chunker, CurrentChunker
from chatchat.rag.retrieval import (
    RetrievalResult,
    Retriever,
    Reranker,
    VectorRetriever,
    BM25Retriever,
    FusionStrategy,
    RRFFusion,
    WeightedFusion,
    NoOpReranker,
    RetrievalStrategy,
    HybridRetriever,
)

__all__ = [
    "RAGQuery",
    "RAGDocument",
    "RAGResponse",
    "RAGServiceInterface",
    "CurrentRAGService",
    "get_rag_service",
    "DocumentParser",
    "CurrentDocumentParser",
    "Chunker",
    "CurrentChunker",
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
]