"""Document Parser 统一层。

M03 仅提供：
- ``interfaces``：DocumentParser 抽象接口
- ``service``：CurrentDocumentParser（复用现有 document_loaders 的 Adapter）
"""

from chatchat.rag.parser.interfaces import DocumentParser
from chatchat.rag.parser.provider import get_document_parser
from chatchat.rag.parser.service import CurrentDocumentParser
from chatchat.rag.parser.service_ragflow import RAGFlowDocumentParser

__all__ = [
    "DocumentParser",
    "CurrentDocumentParser",
    "RAGFlowDocumentParser",
    "get_document_parser",
]