"""Document Parser 统一层测试：Interface / Parse / Metadata / Empty / Invalid。"""

import pytest
from langchain.docstore.document import Document

from chatchat.core.exceptions import ResourceNotFoundError, ValidationError
from chatchat.rag.parser import DocumentParser, CurrentDocumentParser


class TestParserInterface:
    def test_document_parser_is_abstract(self):
        with pytest.raises(TypeError):
            DocumentParser()

    def test_current_parser_implements_interface(self):
        assert issubclass(CurrentDocumentParser, DocumentParser)


class TestParserParse:
    def test_parse_markdown(self, tmp_path):
        md = tmp_path / "sample.md"
        md.write_text("# Title\n\nhello world", encoding="utf-8")
        docs = CurrentDocumentParser().parse(str(md))
        assert isinstance(docs, list)
        assert len(docs) >= 1
        assert all(isinstance(d, Document) for d in docs)
        assert "hello world" in docs[0].page_content

    def test_parse_preserves_source_metadata(self, tmp_path):
        md = tmp_path / "meta.md"
        md.write_text("content", encoding="utf-8")
        docs = CurrentDocumentParser().parse(str(md))
        assert "source" in docs[0].metadata


class TestParserEmpty:
    def test_parse_empty_file_does_not_raise(self, tmp_path):
        md = tmp_path / "empty.md"
        md.write_text("", encoding="utf-8")
        docs = CurrentDocumentParser().parse(str(md))
        assert isinstance(docs, list)


class TestParserInvalid:
    def test_parse_missing_file_raises_resource_not_found(self, tmp_path):
        missing = tmp_path / "nope.md"
        with pytest.raises(ResourceNotFoundError):
            CurrentDocumentParser().parse(str(missing))

    def test_parse_unsupported_extension_raises_validation_error(self, tmp_path):
        f = tmp_path / "data.xyz"
        f.write_text("whatever", encoding="utf-8")
        with pytest.raises(ValidationError):
            CurrentDocumentParser().parse(str(f))