"""当前 Chunker 的 Adapter 实现。

复用现有 ``chatchat.server.knowledge_base.utils.make_text_splitter`` 与
``chatchat.server.file_rag.text_splitter.zh_title_enhance`` 的切分逻辑，
将切分统一收敛到 ``Chunker.chunk`` 接口。本类不复制切分逻辑、
不做 Embedding / VectorStore / Retrieval、不吞异常。
"""

from typing import List, Optional

from langchain.docstore.document import Document

from chatchat.settings import Settings
from chatchat.rag.chunk.interfaces import Chunker
from chatchat.server.knowledge_base.utils import make_text_splitter
from chatchat.server.file_rag.text_splitter import zh_title_enhance as func_zh_title_enhance


class CurrentChunker(Chunker):
    """基于现有 text_splitter 的切分适配器（Adapter）。"""

    def chunk(
        self,
        documents: List[Document],
        splitter_name: Optional[str] = None,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        zh_title_enhance: Optional[bool] = None,
        text_splitter=None,
        skip_split: bool = False,
        **kwargs,
    ) -> List[Document]:
        if not documents:
            return []

        splitter_name = splitter_name or Settings.kb_settings.TEXT_SPLITTER_NAME
        chunk_size = Settings.kb_settings.CHUNK_SIZE if chunk_size is None else chunk_size
        chunk_overlap = Settings.kb_settings.OVERLAP_SIZE if chunk_overlap is None else chunk_overlap
        zh_title_enhance = Settings.kb_settings.ZH_TITLE_ENHANCE if zh_title_enhance is None else zh_title_enhance

        docs = documents
        if not skip_split:
            if text_splitter is None:
                text_splitter = make_text_splitter(
                    splitter_name=splitter_name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            if splitter_name == "MarkdownHeaderTextSplitter":
                docs = text_splitter.split_text(docs[0].page_content)
            else:
                docs = text_splitter.split_documents(docs)

        if not docs:
            return []

        if zh_title_enhance:
            docs = func_zh_title_enhance(docs)
        return docs