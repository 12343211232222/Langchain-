"""Chunk 统一层测试：Interface / Size / Overlap / Metadata / Empty / Long。"""

import pytest
from langchain.docstore.document import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from chatchat.rag.chunk import Chunker, CurrentChunker


def _make_splitter(chunk_size=50, chunk_overlap=10):
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


class TestChunkerInterface:
    def test_chunker_is_abstract(self):
        with pytest.raises(TypeError):
            Chunker()

    def test_current_chunker_implements_interface(self):
        assert issubclass(CurrentChunker, Chunker)


class TestCurrentChunker:
    def test_chunk_splits_long_document(self):
        text = "这是一段测试文本。" * 100
        docs = [Document(page_content=text, metadata={"source": "a.txt"})]
        chunks = CurrentChunker().chunk(
            docs,
            text_splitter=_make_splitter(chunk_size=50, chunk_overlap=10),
            zh_title_enhance=False,
        )
        assert len(chunks) > 1
        assert all(len(c.page_content) <= 50 for c in chunks)

    def test_chunk_overlap_keeps_content_continuity(self):
        text = "abcdefghijklmnopqrstuvwxyz" * 10
        docs = [Document(page_content=text, metadata={})]
        chunks = CurrentChunker().chunk(
            docs,
            text_splitter=_make_splitter(chunk_size=60, chunk_overlap=12),
            zh_title_enhance=False,
        )
        assert len(chunks) > 1
        # 拼接 chunks 应覆盖原文全部内容
        merged = "".join(c.page_content for c in chunks)
        for ch in text:
            assert ch in merged

    def test_chunk_preserves_metadata(self):
        docs = [
            Document(
                page_content="内容" * 100,
                metadata={"source": "s.md", "page": 3, "file_name": "s.md"},
            )
        ]
        chunks = CurrentChunker().chunk(
            docs,
            text_splitter=_make_splitter(),
            zh_title_enhance=False,
        )
        assert chunks
        for c in chunks:
            assert c.metadata["source"] == "s.md"
            assert c.metadata["page"] == 3

    def test_chunk_empty_documents(self):
        assert CurrentChunker().chunk([], zh_title_enhance=False) == []

    def test_chunk_skip_split_returns_documents(self):
        docs = [Document(page_content="不切的文本", metadata={"source": "a.csv"})]
        chunks = CurrentChunker().chunk(docs, skip_split=True, zh_title_enhance=False)
        assert chunks == docs

    def test_chunk_markdown_header_splitter(self):
        text = "# H1\n\n内容一\n\n# H2\n\n内容二"
        docs = [Document(page_content=text, metadata={"source": "b.md"})]
        splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[("#", "h1")], strip_headers=False
        )
        chunks = CurrentChunker().chunk(
            docs,
            splitter_name="MarkdownHeaderTextSplitter",
            text_splitter=splitter,
            zh_title_enhance=False,
        )
        assert len(chunks) == 2
        assert all(c.metadata.get("h1") for c in chunks)

    def test_chunk_long_document_content_integrity(self):
        paragraphs = [f"第{i}段内容。" * 20 for i in range(50)]
        text = "\n\n".join(paragraphs)
        docs = [Document(page_content=text, metadata={"source": "long.txt"})]
        chunks = CurrentChunker().chunk(
            docs,
            text_splitter=_make_splitter(chunk_size=200, chunk_overlap=20),
            zh_title_enhance=False,
        )
        assert len(chunks) > 1
        # 所有 chunk 内容都来自原文
        for c in chunks:
            assert c.page_content.replace("\n\n", "") in text.replace("\n\n", "")