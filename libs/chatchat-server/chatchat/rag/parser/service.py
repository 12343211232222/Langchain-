"""当前 Document Parser 的 Adapter 实现。

复用现有 ``chatchat.server.knowledge_base.utils.get_loader`` 的 Loader 逻辑，
将文件解析统一收敛到 ``DocumentParser.parse`` 接口。本类不复制加载逻辑、
不做 Chunk / Embedding / Retrieval、不吞异常。
"""

import os
from typing import List, Optional

from langchain.docstore.document import Document
from langchain_community.document_loaders import TextLoader

from chatchat.core.exceptions import ResourceNotFoundError, ValidationError
from chatchat.rag.parser.interfaces import DocumentParser
from chatchat.server.knowledge_base.utils import get_loader, get_LoaderClass


class CurrentDocumentParser(DocumentParser):
    """基于现有 document_loaders 的文档解析适配器（Adapter）。"""

    def parse(
        self,
        file_path: str,
        loader_kwargs: Optional[dict] = None,
        **kwargs,
    ) -> List[Document]:
        if not os.path.isfile(file_path):
            raise ResourceNotFoundError(f"文件不存在: {file_path}")

        ext = os.path.splitext(file_path)[-1].lower()
        loader_name = get_LoaderClass(ext)
        if loader_name is None:
            raise ValidationError(f"暂未支持的文件格式: {ext or '(无扩展名)'}")

        loader = get_loader(
            loader_name=loader_name,
            file_path=file_path,
            loader_kwargs=loader_kwargs,
        )
        if isinstance(loader, TextLoader):
            loader.encoding = "utf8"
        return loader.load()