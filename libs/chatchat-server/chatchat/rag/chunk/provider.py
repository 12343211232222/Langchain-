"""Chunker 统一获取入口。

消费方只认 ``Chunker`` 接口；未来接入 RAGFlow Chunker 时
只需在此替换实现，业务代码零修改。
"""

from chatchat.rag.chunk.interfaces import Chunker

_chunker: Chunker | None = None


def get_chunker() -> Chunker:
    """获取当前 Chunker（模块级单例）。"""
    global _chunker
    if _chunker is None:
        from chatchat.rag.chunk.service import CurrentChunker

        _chunker = CurrentChunker()
    return _chunker
