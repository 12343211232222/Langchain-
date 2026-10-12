"""Policy / Guardrail：执行前/后/异常 Hook 抽象。

M07 只建立抽象，不开发完整企业安全系统。
默认策略 ``AllowAllPolicy`` 不改变现有行为。
未来可扩展：输入检查、工具访问控制、Prompt Injection 防护、输出审核等。
"""

from abc import ABC, abstractmethod
from typing import Any, Optional

from chatchat.agent.runtime.result import RuntimeResult
from chatchat.core.logging import get_logger

logger = get_logger(__name__)


class ExecutionPolicy(ABC):
    """执行生命周期 Hook 接口。"""

    @abstractmethod
    def before_execute(self, context: Any) -> None:
        """执行前检查。可抛异常拦截执行。"""
        raise NotImplementedError

    @abstractmethod
    def after_execute(self, context: Any, result: RuntimeResult) -> None:
        """执行后检查。"""
        raise NotImplementedError

    @abstractmethod
    def on_error(self, context: Any, error: str) -> None:
        """执行失败时调用。"""
        raise NotImplementedError


class AllowAllPolicy(ExecutionPolicy):
    """默认放行策略：不拦截任何执行。"""

    def before_execute(self, context: Any) -> None:
        logger.info("policy: allow all (before)")

    def after_execute(self, context: Any, result: RuntimeResult) -> None:
        logger.info(f"policy: allow all (after) success={result.success}")

    def on_error(self, context: Any, error: str) -> None:
        logger.info(f"policy: allow all (on_error) {error}")
