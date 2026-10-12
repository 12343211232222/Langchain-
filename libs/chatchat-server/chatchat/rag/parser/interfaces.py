"""Document Parser 统一接口。

Parser 负责将单个文件解析为 ``langchain.docstore.document.Document`` 列表，
与 Chunk（切分）、Embedding（向量化）、Retrieval（检索）、LLM（生成）彻底解耦。
"""

from abc import ABC, abstractmethod
from typing import List

from langchain.docstore.document import Document


class DocumentParser(ABC):
    """文档解析接口。

    未来无论底层是现有 file_rag/document_loaders、RAGFlow、DeepDoc 还是 OCR，
    上层都通过 ``parse`` 统一调用，输入文件路径，输出标准 Document 列表。
    """

    @abstractmethod
    def parse(self, file_path: str, **kwargs) -> List[Document]:
        """解析文件为 Document 列表。

        :param file_path: 待解析文件的绝对路径
        :param kwargs: 透传给底层 Loader 的参数（如 ``loader_kwargs``）
        :return: 解析出的 Document 列表（metadata 中保留 ``source`` 等原始信息）
        """
        raise NotImplementedError