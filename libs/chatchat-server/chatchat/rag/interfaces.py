"""RAG 服务统一接口。"""

from abc import ABC, abstractmethod

from chatchat.rag.schemas import RAGQuery, RAGResponse


class RAGServiceInterface(ABC):
    """RAG 服务接口。

    未来无论底层是 FAISS / Milvus / RAGFlow，
    上层都通过 ``retrieve`` 统一调用，不依赖具体 VectorStore。
    """

    @abstractmethod
    def retrieve(self, request: RAGQuery) -> RAGResponse:
        """执行检索，仅返回文档列表，不负责 LLM 生成与 Prompt 构造。"""
        raise NotImplementedError