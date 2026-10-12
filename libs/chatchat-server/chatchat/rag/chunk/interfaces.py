"""Chunker 统一接口。

将 Document 列表切分为 Chunk 列表，与 Parser、Embedding、VectorStore 彻底解耦。
"""

from abc import ABC, abstractmethod
from typing import List

from langchain.docstore.document import Document


class Chunker(ABC):
    """文本切分接口。

    未来无论底层是现有 langchain Splitter、RAGFlow Chunker、
    Semantic Chunker 还是 Hierarchical Chunker，上层都通过 ``chunk`` 统一调用。
    """

    @abstractmethod
    def chunk(self, documents: List[Document], **kwargs) -> List[Document]:
        """将 Document 列表切分为 Chunk 列表。

        :param documents: 输入 Document 列表（来自 Parser）
        :param kwargs: 透传给底层 Splitter 的参数（如 ``chunk_size``、``chunk_overlap`` 等）
        :return: 切分后的 Chunk 列表（metadata 保留来源信息）
        """
        raise NotImplementedError