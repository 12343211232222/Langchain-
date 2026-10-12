"""Chunk / Ingestion 统一层。

M04 仅提供：
- ``interfaces``：Chunker 抽象接口
- ``service``：CurrentChunker（复用现有 text_splitter 的 Adapter）
"""

from chatchat.rag.chunk.interfaces import Chunker
from chatchat.rag.chunk.provider import get_chunker
from chatchat.rag.chunk.service import CurrentChunker

__all__ = [
    "Chunker",
    "CurrentChunker",
    "get_chunker",
]