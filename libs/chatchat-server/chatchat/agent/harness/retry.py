"""Retry / Recovery：最小可用重试机制。

当 M06 Runtime 执行失败时，ExecutionManager 会按 ``RetryPolicy``
判断是否可重试、重试次数上限，然后再次调用 Runtime。

注意：M06 ``RuntimeResult.error`` 为字符串（``"TypeError: xxx"`` 格式），
因此 RetryPolicy 用**异常类名字符串匹配**判断可重试性。
"""

from typing import Any, List, Optional, Set

from chatchat.core.logging import get_logger

logger = get_logger(__name__)


class RetryPolicy:
    """最小重试策略。

    参数:
        max_retries: 最大重试次数，默认 0（不重试）。
        retryable_exceptions: 可重试异常类名集合，例如 {"ServiceError", "ConnectionError"}。
    """

    def __init__(
        self,
        max_retries: int = 0,
        retryable_exceptions: Optional[Set[str]] = None,
    ):
        self.max_retries = max_retries
        self.retryable_exceptions: Set[str] = set(retryable_exceptions or {})

    def is_retryable(self, error_text: Optional[str]) -> bool:
        """判断错误文本是否命中可重试异常。"""
        if not error_text:
            return False
        error_name = error_text.split(":")[0].strip()
        return error_name in self.retryable_exceptions

    def should_retry(self, error_text: Optional[str], retry_count: int) -> bool:
        """综合判断是否应再次重试。"""
        if retry_count >= self.max_retries:
            return False
        return self.is_retryable(error_text)

    def record(self, retry_count: int, error_text: Optional[str]) -> None:
        """记录一次重试（仅打日志）。"""
        logger.info(
            f"retry {retry_count}/{self.max_retries}: error={error_text}"
        )
