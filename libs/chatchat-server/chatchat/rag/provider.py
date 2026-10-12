"""RAG Service 统一获取入口。

M02 仅提供 ``CurrentRAGService``；未来可在此扩展为 RAGFlowRAGService 等。
"""

from chatchat.rag.interfaces import RAGServiceInterface
from chatchat.rag.service import CurrentRAGService

_rag_service: RAGServiceInterface | None = None


def get_rag_service() -> RAGServiceInterface:
    """获取当前 RAG Service（模块级单例）。"""
    global _rag_service
    if _rag_service is None:
        _rag_service = CurrentRAGService()
    return _rag_service