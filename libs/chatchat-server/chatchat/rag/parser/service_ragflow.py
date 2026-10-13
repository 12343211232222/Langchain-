"""RAGFlow DeepDoc 解析器实现（M03 扩展）。

RAGFlow 在本任务中只扮演「把文件解析成结构化文本的函数」：
提交 → 轮询 → 拉取 → 清理，解析完成即删除临时 dataset，
不在 RAGFlow 侧留存任何数据，也不引入其检索能力。

明确不做：
- 不替换 Chunker（返回的 chunk 直接当 Document 输出，后续仍由
  ``CurrentChunker`` 再处理一次——既定设计）
- 不上传到持久 dataset / 不维护双份知识库
- 不影响 ``CurrentDocumentParser``（它仍是默认与回退实现）

注意：RAGFlow 不同版本接口路径与字段有差异（老版单数
``/api/v1/dataset/...``，新版复数 ``/api/v1/datasets/...``），
本实现按新版复数路径编写，对接实际部署版本时以官方文档为准。

大批量场景健壮性（本模块增强）：
- ``_semaphore`` 是**类级**信号量：限制同时进行的 RAGFlow 解析会话数。
  之所以放在类级而非实例级，是因为 provider 单例可能被多线程共享，
  实例级信号量无法限制全局并发；RAGFlow ingestor 通常只有 1 个 worker，
  并发请求只会在其内部排队并集体撞上超时，所以闸门设在客户端侧。
- ``cleanup_orphan_datasets``：进程被杀 / finally 未执行时残留的
  ``cc_temp_parse_*`` 临时库，通过此方法按前缀 + create_time 兜底回收。
"""

import os
import threading
import time
import uuid
from typing import List, Optional

import httpx
from langchain.docstore.document import Document

from chatchat.core.exceptions import (
    ConfigurationError,
    ResourceNotFoundError,
    ServiceError,
)
from chatchat.core.logging import get_logger
from chatchat.rag.parser.interfaces import DocumentParser
from chatchat.settings import Settings

logger = get_logger(__name__)

# 轮询期间容忍的连续网络异常次数。轮询间隔很小（默认 0.5s），
# 偶发网络抖动不应导致整个解析失败；连续超过该次数才判定为服务不可用。
RAGFLOW_POLL_MAX_ERRORS = 3


class RAGFlowDocumentParser(DocumentParser):
    """基于 RAGFlow DeepDoc 的 Document Parser（M03）。

    每次 parse：建临时 dataset → 上传 → 触发解析 → 轮询 → 拉取 chunks
    → 转 Document → finally 删除临时 dataset。
    """

    # 类级并发闸门：所有实例共享，进程内全局生效。
    # 容量在首次 parse 时按 RAGFLOW_MAX_CONCURRENT 惰性创建（创建后不可变）。
    _semaphore: Optional[threading.Semaphore] = None
    _semaphore_lock = threading.Lock()

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        poll_interval: Optional[float] = None,
        parse_timeout: Optional[float] = None,
        temp_dataset_prefix: Optional[str] = None,
    ):
        """参数默认读取 ``Settings.basic_settings.RAGFLOW_*``。"""
        cfg = Settings.basic_settings
        self.base_url = (base_url if base_url is not None else cfg.RAGFLOW_BASE_URL).rstrip("/")
        self.api_key = api_key if api_key is not None else cfg.RAGFLOW_API_KEY
        self.poll_interval = (
            poll_interval if poll_interval is not None else cfg.RAGFLOW_PARSE_POLL_INTERVAL
        )
        self.parse_timeout = (
            parse_timeout if parse_timeout is not None else cfg.RAGFLOW_PARSE_TIMEOUT
        )
        self.temp_dataset_prefix = (
            temp_dataset_prefix
            if temp_dataset_prefix is not None
            else cfg.RAGFLOW_TEMP_DATASET_PREFIX
        )
        if not self.base_url or not self.api_key:
            raise ConfigurationError(
                "RAGFlow Parser 未配置 RAGFLOW_BASE_URL / RAGFLOW_API_KEY，"
                "请在 basic_settings.yaml 中填写后再启用 RAGFLOW_PARSER_ENABLED"
            )

    # ---- 对外接口 ----

    @classmethod
    def _get_semaphore(cls) -> threading.Semaphore:
        """惰性创建类级信号量（双检锁，线程安全）。"""
        if cls._semaphore is None:
            with cls._semaphore_lock:
                if cls._semaphore is None:
                    n = int(Settings.basic_settings.RAGFLOW_MAX_CONCURRENT)
                    cls._semaphore = threading.Semaphore(max(1, n))
        return cls._semaphore

    def parse(self, file_path: str, **kwargs) -> List[Document]:
        """解析文件为 Document 列表（同步封装异步解析流程）。

        信号量保护整个「建库→上传→解析→拉取→清理」流程；
        先 acquire 再进入 finally 结构，保证 release 一定在
        临时 dataset 清理之后执行。
        """
        if not os.path.isfile(file_path):
            raise ResourceNotFoundError(f"文件不存在: {file_path}")

        semaphore = self._get_semaphore()
        with semaphore:
            dataset_id: Optional[str] = None
            with self._client() as client:
                try:
                    dataset_id = self._create_temp_dataset(client)
                    document_id = self._upload_document(client, dataset_id, file_path)
                    self._trigger_parse(client, dataset_id, document_id)
                    self._wait_until_done(client, dataset_id, document_id)
                    chunks = self._fetch_chunks(client, dataset_id, document_id)
                    return [self._to_document(c, file_path) for c in chunks]
                finally:
                    if dataset_id:
                        self._delete_dataset(client, dataset_id)

    # ---- 内部步骤 ----

    def _client(self) -> httpx.Client:
        return httpx.Client(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=float(Settings.basic_settings.HTTPX_DEFAULT_TIMEOUT),
        )

    @staticmethod
    def _unwrap(resp: httpx.Response, step: str):
        """统一处理 RAGFlow 响应格式：{"code": 0, "data": ...}。"""
        if resp.status_code == 401:
            raise ServiceError(f"RAGFlow {step} 鉴权失败（HTTP 401），请检查 RAGFLOW_API_KEY")
        try:
            payload = resp.json()
        except ValueError as e:
            raise ServiceError(f"RAGFlow {step} 返回非 JSON（HTTP {resp.status_code}）: {e}")
        code = payload.get("code")
        if code != 0:
            raise ServiceError(
                f"RAGFlow {step} 失败（code={code}）: {payload.get('message')}"
            )
        return payload.get("data")

    def _create_temp_dataset(self, client: httpx.Client) -> str:
        name = f"{self.temp_dataset_prefix}{uuid.uuid4().hex[:12]}"
        data = self._unwrap(
            client.post("/api/v1/datasets", json={"name": name}), "创建临时 dataset"
        )
        dataset_id = data.get("id")
        if not dataset_id:
            raise ServiceError("RAGFlow 创建临时 dataset 未返回 id")
        logger.info(f"ragflow temp dataset created: {dataset_id}")
        return dataset_id

    def _upload_document(self, client: httpx.Client, dataset_id: str, file_path: str) -> str:
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            data = self._unwrap(
                client.post(
                    f"/api/v1/datasets/{dataset_id}/documents", files=files
                ),
                "上传文件",
            )
        docs = data if isinstance(data, list) else [data]
        if not docs or not docs[0].get("id"):
            raise ServiceError("RAGFlow 上传文件未返回 document id")
        return docs[0]["id"]

    def _trigger_parse(self, client: httpx.Client, dataset_id: str, document_id: str) -> None:
        self._unwrap(
            client.post(
                f"/api/v1/datasets/{dataset_id}/chunks",
                json={"document_ids": [document_id]},
            ),
            "触发解析",
        )

    def _wait_until_done(self, client: httpx.Client, dataset_id: str, document_id: str) -> None:
        """轮询解析进度，直到 DONE / 失败 / 超时。

        容错设计：轮询间隔调小后（默认 0.5s）轮询次数大幅增加，偶发的网络抖动
        不应让整个解析前功尽弃。因此对单次轮询的网络异常（``httpx.TransportError``）
        容忍连续若干次，超过阈值才向上抛；而业务失败（run=FAIL/CANCEL）仍立即抛。
        """
        deadline = time.monotonic() + self.parse_timeout
        consecutive_errors = 0
        last_error: Optional[Exception] = None
        while True:
            try:
                data = self._unwrap(
                    client.get(
                        f"/api/v1/datasets/{dataset_id}/documents",
                        params={"id": document_id},
                    ),
                    "轮询解析进度",
                )
                consecutive_errors = 0  # 本轮成功，重置容错计数
            except httpx.TransportError as e:
                # 网络层抖动：容忍连续 N 次，期间继续按 deadline 重试
                consecutive_errors += 1
                last_error = e
                if consecutive_errors >= RAGFLOW_POLL_MAX_ERRORS:
                    raise ServiceError(
                        f"RAGFlow 轮询连续失败 {consecutive_errors} 次"
                        f"（已等待 {self.parse_timeout - (deadline - time.monotonic()):.1f}s）: {e}"
                    ) from e
                logger.warning(
                    f"ragflow 轮询异常（第 {consecutive_errors}/{RAGFLOW_POLL_MAX_ERRORS} 次，"
                    f"将继续重试）: {e}"
                )
                if time.monotonic() >= deadline:
                    break
                time.sleep(self.poll_interval)
                continue

            docs = (data or {}).get("docs") or []
            doc = next((d for d in docs if d.get("id") == document_id), docs[0] if docs else None)
            if doc is not None:
                run = str(doc.get("run", "")).upper()
                progress = doc.get("progress", 0) or 0
                if run == "DONE" or progress >= 1:
                    return
                if run in ("FAIL", "CANCEL"):
                    raise ServiceError(
                        f"RAGFlow 解析失败（run={run}）: {doc.get('progress_msg')}"
                    )
            if time.monotonic() >= deadline:
                break
            time.sleep(self.poll_interval)

        # 超时退出：错误信息带「已等待时长 / 配置上限」，便于排查
        waited = self.parse_timeout - (deadline - time.monotonic())
        detail = f"，最后一次轮询异常: {last_error}" if last_error else ""
        raise ServiceError(
            f"RAGFlow 解析超时（已等待 {waited:.1f}s / 上限 {self.parse_timeout}s）"
            f": {document_id}{detail}"
        )

    def _fetch_chunks(
        self, client: httpx.Client, dataset_id: str, document_id: str
    ) -> list:
        """拉取解析结果。

        注意：RAGFlow 的 progress 到 1 时 chunk 可能尚未完全落库（v1.0.0-rc1 实测
        存在数秒写入延迟），因此空结果时短暂重试，避免竞态导致误判为空文档。
        """
        deadline = time.monotonic() + 30.0
        attempts = 0
        while True:
            attempts += 1
            data = self._unwrap(
                client.get(
                    f"/api/v1/datasets/{dataset_id}/documents/{document_id}/chunks",
                    params={"page_size": 1024},
                ),
                "拉取解析结果",
            )
            chunks = (data or {}).get("chunks") or []
            if chunks:
                logger.info(f"ragflow chunks fetched: {len(chunks)} (attempt {attempts})")
                return chunks
            if time.monotonic() >= deadline:
                logger.warning(
                    f"ragflow chunks 为空（已重试 {attempts} 次 / 30s），"
                    f"按 0 个 chunk 处理: {document_id}"
                )
                return []
            time.sleep(2.0)

    @staticmethod
    def _to_document(chunk: dict, file_path: str) -> Document:
        content = chunk.get("content_with_weight") or chunk.get("content") or ""
        metadata = {"source": file_path}
        for key in ("document_keyword", "similarity"):
            if chunk.get(key) is not None:
                metadata[key] = chunk[key]
        return Document(page_content=content, metadata=metadata)

    def _delete_dataset(self, client: httpx.Client, dataset_id: str) -> None:
        """清理临时 dataset；失败只记日志，不覆盖主流程异常。"""
        try:
            self._unwrap(
                client.request("DELETE", "/api/v1/datasets", json={"ids": [dataset_id]}),
                "删除临时 dataset",
            )
            logger.info(f"ragflow temp dataset deleted: {dataset_id}")
        except Exception as e:
            logger.error(f"ragflow 临时 dataset 清理失败 {dataset_id}: {e}")

    # ---- 孤儿临时库兜底清理 ----

    def cleanup_orphan_datasets(self, older_than_seconds: int = 3600) -> int:
        """回收异常残留的临时 dataset（进程被杀 / finally 未执行的场景）。

        安全约束：只删除 name 以 ``self.temp_dataset_prefix`` 开头且超龄的库，
        绝不碰其他 dataset。返回清理数量。

        时间字段说明（v1.0.0-rc1 实测）：列表接口 ``GET /api/v1/datasets``
        **不返回 ``create_time``**（该字段仅在创建响应里有），只返回
        ``update_time``（毫秒时间戳）。临时库的寿命极短（建库即解析即删除），
        因此 ``update_time ≈ create_time``，用前者作兜底判断是安全的。
        两个字段都缺失时才退化为「只报告不删除」。
        """
        with self._client() as client:
            data = self._unwrap(
                client.get("/api/v1/datasets", params={"page_size": 1024}),
                "列出 datasets",
            )
            datasets = data if isinstance(data, list) else (data or {}).get("datasets") or []
            orphans = []
            now_ms = int(time.time() * 1000)
            missing_time = 0
            for ds in datasets:
                name = ds.get("name") or ""
                if not name.startswith(self.temp_dataset_prefix):
                    continue  # 安全红线：非本模块创建的库一律跳过
                # 优先 create_time，回退 update_time（rc1 列表接口只有后者）
                ts = ds.get("create_time")
                if ts is None:
                    ts = ds.get("update_time")
                if ts is None:
                    missing_time += 1
                    continue
                age_seconds = (now_ms - int(ts)) / 1000.0
                if age_seconds >= older_than_seconds:
                    orphans.append(ds["id"])
            if missing_time:
                logger.warning(
                    f"ragflow {missing_time} 个临时库缺少 create_time/update_time，"
                    "只报告不删除（请升级 RAGFlow 或手工核对）"
                )
            if not orphans:
                logger.info("ragflow 无超龄临时库需要清理")
                return 0
            self._unwrap(
                client.request("DELETE", "/api/v1/datasets", json={"ids": orphans}),
                "清理孤儿临时库",
            )
            for oid in orphans:
                logger.info(f"ragflow orphan dataset cleaned: {oid}")
            return len(orphans)
