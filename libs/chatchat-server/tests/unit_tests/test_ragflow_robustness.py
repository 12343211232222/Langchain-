"""RAGFlow Parser 大批量场景健壮性测试。

覆盖：类级信号量并发闸门、孤儿临时库兜底清理、超时路径快速释放。
"""

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from chatchat.core.exceptions import ServiceError
from chatchat.rag.parser.service_ragflow import RAGFlowDocumentParser
from tests.unit_tests.test_ragflow_parser import (
    DATASET_ID,
    DOC_ID,
    FakeClient,
    FakeResponse,
    make_routes,
)


@pytest.fixture(autouse=True)
def reset_semaphore(monkeypatch):
    """每个用例前后重置类级信号量，避免容量跨用例污染。"""
    monkeypatch.setattr(RAGFlowDocumentParser, "_semaphore", None)
    yield
    monkeypatch.setattr(RAGFlowDocumentParser, "_semaphore", None)


@pytest.fixture
def parser_factory(monkeypatch):
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


class TestSemaphore:
    def test_class_level_singleton(self, parser_factory):
        p1, _ = parser_factory()
        p2, _ = parser_factory()
        s1 = RAGFlowDocumentParser._get_semaphore()
        s2 = RAGFlowDocumentParser._get_semaphore()
        assert s1 is s2  # 类级单例，不是每实例各持一个

    def test_concurrency_limited(self, parser_factory, monkeypatch, tmp_path):
        """并发 parse() 时，同时进入临界区的数量不超过 RAGFLOW_MAX_CONCURRENT。"""
        from chatchat.settings import Settings

        limit = 2
        monkeypatch.setattr(Settings.basic_settings, "RAGFLOW_MAX_CONCURRENT", limit)

        active = 0
        peak = 0
        lock = threading.Lock()

        def slow_wait(client, dataset_id, document_id):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.2)  # 模拟解析耗时，放大并发窗口
            with lock:
                active -= 1

        f = tmp_path / "t.pdf"
        f.write_bytes(b"x")
        routes = make_routes()
        # 同一 parser（共享类级信号量），但每线程需要各自 client
        p, _ = parser_factory(routes=routes)
        monkeypatch.setattr(p, "_wait_until_done", slow_wait)
        monkeypatch.setattr(p, "_client", lambda: FakeClient(make_routes()))

        with ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(lambda _: p.parse(str(f)), range(6)))

        assert all(len(r) == 2 for r in results)
        assert peak <= limit
        assert peak == limit  # 闸门确实被顶满过，证明测试有效

    def test_release_on_failure(self, parser_factory, tmp_path):
        """异常路径信号量也要释放（with 语句保证），后续调用不被卡死。"""
        f = tmp_path / "t.pdf"
        f.write_bytes(b"x")
        routes = make_routes()
        routes[("POST", f"/api/v1/datasets/{DATASET_ID}/chunks")] = {
            "code": 102,
            "message": "parse failed",
        }
        p, _ = parser_factory(routes=routes)
        with pytest.raises(ServiceError):
            p.parse(str(f))
        sem = RAGFlowDocumentParser._get_semaphore()
        # 容量未被泄漏：acquire 非阻塞应成功
        assert sem.acquire(blocking=False)
        sem.release()


class TestCleanupOrphanDatasets:
    NOW_MS = int(time.time() * 1000)

    def _routes(self, datasets):
        return {
            ("GET", "/api/v1/datasets"): {"code": 0, "data": datasets},
            ("DELETE", "/api/v1/datasets"): {"code": 0, "data": True},
        }

    def test_only_prefix_matched_and_aged_deleted(self, parser_factory):
        datasets = [
            # 2 个前缀匹配且超龄 → 应删
            {"id": "o1", "name": "cc_temp_parse_aaa", "create_time": self.NOW_MS - 7200_000},
            {"id": "o2", "name": "cc_temp_parse_bbb", "create_time": self.NOW_MS - 4000_000},
            # 前缀匹配但未超龄 → 保留
            {"id": "y1", "name": "cc_temp_parse_ccc", "create_time": self.NOW_MS - 60_000},
            # 其他名字 → 绝不碰
            {"id": "x1", "name": "my_kb", "create_time": self.NOW_MS - 99999_000},
        ]
        p, client = parser_factory(routes=self._routes(datasets))
        count = p.cleanup_orphan_datasets(older_than_seconds=3600)
        assert count == 2
        delete_calls = [c for c in client.calls if c[0] == "DELETE"]
        assert len(delete_calls) == 1
        assert set(delete_calls[0][2]["json"]["ids"]) == {"o1", "o2"}

    def test_nothing_deleted_when_no_prefix_match(self, parser_factory):
        datasets = [
            {"id": "x1", "name": "kb_a", "create_time": self.NOW_MS - 99999_000},
            {"id": "x2", "name": "kb_b", "create_time": self.NOW_MS - 99999_000},
        ]
        p, client = parser_factory(routes=self._routes(datasets))
        count = p.cleanup_orphan_datasets(older_than_seconds=3600)
        assert count == 0
        assert not any(c[0] == "DELETE" for c in client.calls)

    def test_missing_create_time_reports_only(self, parser_factory):
        datasets = [
            {"id": "o1", "name": "cc_temp_parse_aaa"},  # 两个时间字段都没有
        ]
        p, client = parser_factory(routes=self._routes(datasets))
        count = p.cleanup_orphan_datasets(older_than_seconds=3600)
        assert count == 0  # 退化为只报告不删除
        assert not any(c[0] == "DELETE" for c in client.calls)

    def test_fallback_to_update_time(self, parser_factory):
        """v1.0.0-rc1 列表接口不返回 create_time，只用 update_time 也应能清理。"""
        datasets = [
            # 有 update_time、无 create_time，且超龄 → 应删
            {"id": "u1", "name": "cc_temp_parse_uuu", "update_time": self.NOW_MS - 7200_000},
            # 有 update_time 但未超龄 → 保留
            {"id": "u2", "name": "cc_temp_parse_vvv", "update_time": self.NOW_MS - 60_000},
            # 两个字段都有，create_time 为准
            {
                "id": "u3",
                "name": "cc_temp_parse_www",
                "create_time": self.NOW_MS - 7200_000,
                "update_time": self.NOW_MS - 1_000,
            },
        ]
        p, client = parser_factory(routes=self._routes(datasets))
        count = p.cleanup_orphan_datasets(older_than_seconds=3600)
        assert count == 2
        delete_calls = [c for c in client.calls if c[0] == "DELETE"]
        assert set(delete_calls[0][2]["json"]["ids"]) == {"u1", "u3"}


class TestTimeoutFastRelease:
    def test_timeout_deletes_temp_dataset(self, parser_factory, tmp_path):
        """轮询永远 progress=0.5 → 超时抛 ServiceError，且 finally 清理临时库。"""
        f = tmp_path / "t.pdf"
        f.write_bytes(b"x")
        routes = make_routes(
            poll_payload={
                "code": 0,
                "data": {"docs": [{"id": DOC_ID, "run": "RUNNING", "progress": 0.5}]},
            }
        )
        p, client = parser_factory(routes=routes, parse_timeout=0.01)
        with pytest.raises(ServiceError, match="解析超时"):
            p.parse(str(f))
        delete_calls = [c for c in client.calls if c[0] == "DELETE"]
        assert len(delete_calls) == 1
        assert delete_calls[0][2]["json"] == {"ids": [DATASET_ID]}

    def test_timeout_message_contains_waited_and_limit(self, parser_factory, tmp_path):
        """超时错误信息必须同时包含「已等待时长」与「配置上限」，便于排查。"""
        f = tmp_path / "t.pdf"
        f.write_bytes(b"x")
        routes = make_routes(
            poll_payload={
                "code": 0,
                "data": {"docs": [{"id": DOC_ID, "run": "RUNNING", "progress": 0.5}]},
            }
        )
        p, _ = parser_factory(routes=routes, parse_timeout=0.01)
        with pytest.raises(ServiceError) as ei:
            p.parse(str(f))
        msg = str(ei.value)
        assert "已等待" in msg
        assert "上限" in msg
        assert "0.01" in msg  # 实际超时配置值


class TestPollErrorTolerance:
    def test_transient_errors_are_tolerated(self, parser_factory, tmp_path, monkeypatch):
        """轮询前 2 次抛网络异常、第 3 次返回 DONE → 应容忍并最终成功。"""
        import httpx

        f = tmp_path / "t.pdf"
        f.write_bytes(b"x")
        routes = make_routes()
        state = {"n": 0}

        def flaky_poll(**kwargs):
            state["n"] += 1
            if state["n"] <= 2:
                raise httpx.ConnectError("模拟网络抖动")
            return FakeResponse(
                {"code": 0, "data": {"docs": [{"id": DOC_ID, "run": "DONE", "progress": 1.0}]}}
            )

        routes[("GET", f"/api/v1/datasets/{DATASET_ID}/documents")] = flaky_poll
        p, _ = parser_factory(routes=routes, parse_timeout=30)
        docs = p.parse(str(f))
        assert len(docs) == 2  # 最终成功拿到解析结果
        assert state["n"] == 3  # 确实重试了

    def test_persistent_errors_raise_after_threshold(self, parser_factory, tmp_path):
        """网络异常持续超过阈值 → 抛 ServiceError，避免无限重试。"""
        import httpx

        from chatchat.rag.parser.service_ragflow import RAGFLOW_POLL_MAX_ERRORS

        f = tmp_path / "t.pdf"
        f.write_bytes(b"x")
        routes = make_routes()

        def always_fail(**kwargs):
            raise httpx.ConnectError("持续不可达")

        routes[("GET", f"/api/v1/datasets/{DATASET_ID}/documents")] = always_fail
        p, _ = parser_factory(routes=routes, parse_timeout=30)
        with pytest.raises(ServiceError, match=f"连续失败 {RAGFLOW_POLL_MAX_ERRORS} 次"):
            p.parse(str(f))

    def test_errors_are_reset_by_success(self, parser_factory, tmp_path):
        """成功一次后，容错计数应重置（避免累计误判）。"""
        import httpx

        f = tmp_path / "t.pdf"
        f.write_bytes(b"x")
        routes = make_routes()
        state = {"n": 0}

        def flaky_poll(**kwargs):
            state["n"] += 1
            # 交替：失败、失败、成功、失败、失败、成功 …
            if state["n"] % 3 != 0:
                raise httpx.ConnectError("抖动")
            return FakeResponse(
                {"code": 0, "data": {"docs": [{"id": DOC_ID, "run": "RUNNING", "progress": 0.5}]}}
            )

        routes[("GET", f"/api/v1/datasets/{DATASET_ID}/documents")] = flaky_poll
        # 每次最多连续失败 2 次就会成功一次 → 不会触发阈值（3），最终因超时退出
        p, _ = parser_factory(routes=routes, parse_timeout=0.05)
        with pytest.raises(ServiceError, match="解析超时"):
            p.parse(str(f))
        assert state["n"] >= 3  # 经历了多轮"失败-失败-成功"，未被阈值提前打断
