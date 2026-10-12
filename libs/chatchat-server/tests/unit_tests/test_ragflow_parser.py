"""RAGFlowDocumentParser 单元测试（M03 扩展）。

全部打桩 httpx.Client，不发起真实网络请求。
"""

import pytest

from chatchat.core.exceptions import (
    ConfigurationError,
    ResourceNotFoundError,
    ServiceError,
)
from chatchat.rag.parser.service_ragflow import RAGFlowDocumentParser


# ---------- 打桩基础设施 ----------


class FakeResponse:
    def __init__(self, payload, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def json(self):
        return self._payload


class FakeClient:
    """按 (method, url) 路由返回预置响应，记录所有调用。"""

    def __init__(self, routes: dict):
        self.routes = routes
        self.calls = []  # (method, url, kwargs)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def _dispatch(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        handler = self.routes.get((method, url))
        if handler is None:
            return FakeResponse({"code": 102, "message": f"未打桩: {method} {url}"})
        if isinstance(handler, FakeResponse):
            return handler
        if callable(handler):
            return handler(**kwargs)
        return FakeResponse(handler)

    def get(self, url, **kwargs):
        return self._dispatch("GET", url, **kwargs)

    def post(self, url, **kwargs):
        return self._dispatch("POST", url, **kwargs)

    def request(self, method, url, **kwargs):
        return self._dispatch(method, url, **kwargs)


DATASET_ID = "ds_1"
DOC_ID = "doc_1"


def make_routes(chunks=None, run="DONE", poll_payload=None):
    """构造完整成功流程的路由表。"""
    chunks = chunks if chunks is not None else [
        {
            "content": "备用内容",
            "content_with_weight": "第一条解析内容",
            "document_keyword": "test.pdf",
            "similarity": 0.9,
        },
        {"content": "第二条解析内容"},
    ]
    poll = poll_payload or {
        "code": 0,
        "data": {"docs": [{"id": DOC_ID, "run": run, "progress": 1.0}]},
    }
    return {
        ("POST", "/api/v1/datasets"): {"code": 0, "data": {"id": DATASET_ID}},
        ("POST", f"/api/v1/datasets/{DATASET_ID}/documents"): {
            "code": 0,
            "data": [{"id": DOC_ID}],
        },
        ("POST", f"/api/v1/datasets/{DATASET_ID}/chunks"): {"code": 0, "data": True},
        ("GET", f"/api/v1/datasets/{DATASET_ID}/documents"): poll,
        (
            "GET",
            f"/api/v1/datasets/{DATASET_ID}/documents/{DOC_ID}/chunks",
        ): {"code": 0, "data": {"chunks": chunks}},
        ("DELETE", "/api/v1/datasets"): {"code": 0, "data": True},
    }


@pytest.fixture
def parser_factory(monkeypatch, tmp_path):
    """返回 (parser, client)；client 可通过 routes 参数自定义。"""

    def factory(routes=None, **kwargs):
        routes = routes if routes is not None else make_routes()
        client = FakeClient(routes)
        kwargs.setdefault("base_url", "http://ragflow.test")
        kwargs.setdefault("api_key", "test-key")
        kwargs.setdefault("poll_interval", 0)
        p = RAGFlowDocumentParser(**kwargs)
        monkeypatch.setattr(p, "_client", lambda: client)
        return p, client

    return factory


# ---------- 测试 ----------


class TestParseValidation:
    def test_file_not_found(self, parser_factory):
        p, _ = parser_factory()
        with pytest.raises(ResourceNotFoundError, match="文件不存在"):
            p.parse("C:/no/such/file.pdf")

    def test_missing_config_raises(self, monkeypatch):
        from chatchat.settings import Settings

        monkeypatch.setattr(Settings.basic_settings, "RAGFLOW_BASE_URL", "")
        monkeypatch.setattr(Settings.basic_settings, "RAGFLOW_API_KEY", "")
        with pytest.raises(ConfigurationError):
            RAGFlowDocumentParser()


class TestHappyPath:
    def test_parse_success(self, parser_factory, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_bytes(b"fake pdf")
        p, client = parser_factory()

        docs = p.parse(str(f))

        assert len(docs) == 2
        # content_with_weight 优先
        assert docs[0].page_content == "第一条解析内容"
        assert docs[1].page_content == "第二条解析内容"
        # metadata.source 与原始文件路径一致
        assert docs[0].metadata["source"] == str(f)
        assert docs[0].metadata["document_keyword"] == "test.pdf"
        assert docs[0].metadata["similarity"] == 0.9
        # 流程完整：建 → 传 → 解析 → 轮询 → 拉取 → 删
        methods = [c[0] for c in client.calls]
        assert methods[:1] == ["POST"] and client.calls[0][1] == "/api/v1/datasets"
        assert ("DELETE", "/api/v1/datasets") in [(m, u) for m, u, _ in client.calls]

    def test_temp_dataset_deleted_on_success(self, parser_factory, tmp_path):
        f = tmp_path / "a.txt"
        f.write_text("x")
        p, client = parser_factory()
        p.parse(str(f))
        delete_calls = [c for c in client.calls if c[0] == "DELETE"]
        assert len(delete_calls) == 1
        assert delete_calls[0][2]["json"] == {"ids": [DATASET_ID]}


class TestFailurePaths:
    def test_poll_timeout(self, parser_factory, tmp_path):
        f = tmp_path / "t.pdf"
        f.write_bytes(b"x")
        routes = make_routes(
            poll_payload={
                "code": 0,
                "data": {"docs": [{"id": DOC_ID, "run": "RUNNING", "progress": 0.5}]},
            }
        )
        p, _ = parser_factory(routes=routes, parse_timeout=0.01)
        with pytest.raises(ServiceError, match="解析超时"):
            p.parse(str(f))

    def test_nonzero_code_raises_service_error(self, parser_factory, tmp_path):
        f = tmp_path / "t.pdf"
        f.write_bytes(b"x")
        routes = make_routes()
        routes[("POST", f"/api/v1/datasets/{DATASET_ID}/documents")] = {
            "code": 102,
            "message": "unsupported file type",
        }
        p, _ = parser_factory(routes=routes)
        with pytest.raises(ServiceError, match="unsupported file type"):
            p.parse(str(f))

    def test_dataset_cleaned_up_on_error(self, parser_factory, tmp_path):
        f = tmp_path / "t.pdf"
        f.write_bytes(b"x")
        routes = make_routes()
        routes[("POST", f"/api/v1/datasets/{DATASET_ID}/chunks")] = {
            "code": 102,
            "message": "parse failed",
        }
        p, client = parser_factory(routes=routes)
        with pytest.raises(ServiceError):
            p.parse(str(f))
        # 异常路径下临时 dataset 仍被清理
        assert any(c[0] == "DELETE" for c in client.calls)

    def test_auth_failure(self, parser_factory, tmp_path):
        f = tmp_path / "t.pdf"
        f.write_bytes(b"x")
        routes = make_routes()
        routes[("POST", "/api/v1/datasets")] = FakeResponse(
            {"code": 401, "message": "unauthorized"}, status_code=401
        )
        p, _ = parser_factory(routes=routes)
        with pytest.raises(ServiceError, match="鉴权失败"):
            p.parse(str(f))


class TestProviderSwitch:
    @pytest.fixture(autouse=True)
    def reset_singleton(self, monkeypatch):
        import chatchat.rag.parser.provider as provider_mod

        monkeypatch.setattr(provider_mod, "_parser", None)
        yield
        monkeypatch.setattr(provider_mod, "_parser", None)

    def _set(self, monkeypatch, **cfg):
        from chatchat.settings import Settings

        for k, v in cfg.items():
            monkeypatch.setattr(Settings.basic_settings, k, v)

    def test_disabled_returns_current(self, monkeypatch):
        self._set(monkeypatch, RAGFLOW_PARSER_ENABLED=False)
        from chatchat.rag.parser import CurrentDocumentParser, get_document_parser

        assert isinstance(get_document_parser(), CurrentDocumentParser)

    def test_enabled_returns_ragflow(self, monkeypatch):
        self._set(
            monkeypatch,
            RAGFLOW_PARSER_ENABLED=True,
            RAGFLOW_BASE_URL="http://ragflow.test",
            RAGFLOW_API_KEY="k",
        )
        from chatchat.rag.parser import RAGFlowDocumentParser, get_document_parser

        assert isinstance(get_document_parser(), RAGFlowDocumentParser)

    def test_enabled_but_unconfigured_falls_back(self, monkeypatch):
        self._set(
            monkeypatch,
            RAGFLOW_PARSER_ENABLED=True,
            RAGFLOW_BASE_URL="",
            RAGFLOW_API_KEY="",
        )
        from chatchat.rag.parser import CurrentDocumentParser, get_document_parser

        assert isinstance(get_document_parser(), CurrentDocumentParser)

    def test_unknown_value_falls_back(self, monkeypatch):
        self._set(monkeypatch, RAGFLOW_PARSER_ENABLED="not-a-bool")
        from chatchat.rag.parser import CurrentDocumentParser, get_document_parser

        assert isinstance(get_document_parser(), CurrentDocumentParser)

    def test_singleton(self, monkeypatch):
        self._set(monkeypatch, RAGFLOW_PARSER_ENABLED=False)
        from chatchat.rag.parser import get_document_parser

        assert get_document_parser() is get_document_parser()
