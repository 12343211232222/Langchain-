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
"""

import os
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


class RAGFlowDocumentParser(DocumentParser):
    """基于 RAGFlow DeepDoc 的 Document Parser（M03）。

    每次 parse：建临时 dataset → 上传 → 触发解析 → 轮询 → 拉取 chunks
    → 转 Document → finally 删除临时 dataset。
    """

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

    def parse(self, file_path: str, **kwargs) -> List[Document]:
        """解析文件为 Document 列表（同步封装异步解析流程）。"""
        if not os.path.isfile(file_path):
            raise ResourceNotFoundError(f"文件不存在: {file_path}")

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
        deadline = time.monotonic() + self.parse_timeout
        while True:
            data = self._unwrap(
                client.get(
                    f"/api/v1/datasets/{dataset_id}/documents",
                    params={"id": document_id},
                ),
                "轮询解析进度",
            )
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
                raise ServiceError(
                    f"RAGFlow 解析超时（>{self.parse_timeout}s）: {document_id}"
                )
            time.sleep(self.poll_interval)

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
