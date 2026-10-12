"""Execution Manager：统一控制一次 Agent 执行生命周期。

职责：
1. 用 Context 构建 Runtime 输入
2. 调用 M06 Runtime
3. 获取 RuntimeResult
4. 更新 Context
5. Retry / Recovery
6. 返回 HarnessResult（含 ExecutionMetadata）

Policy hooks 与 Memory save 由上层 ``DefaultAgentHarness`` 编排，
ExecutionManager 不重新实现 Graph，真正执行仍由 M06 ``LangGraphRuntime`` 负责。
"""

import time
from typing import Any, Dict, Optional

from chatchat.agent.harness.context import HarnessContext
from chatchat.agent.harness.result import HarnessResult
from chatchat.agent.harness.retry import RetryPolicy
from chatchat.agent.runtime import AgentRuntimeInterface, RuntimeResult
from chatchat.core.logging import get_logger

logger = get_logger(__name__)


class ExecutionMetadata(Dict[str, Any]):
    """执行元数据（最小 Observability，见 M07 规格）。"""


class ExecutionManager:
    """Execution Manager 实现。

    参数:
        runtime: M06 ``AgentRuntimeInterface``（如 ``LangGraphRuntime``）。
        retry_policy: ``RetryPolicy``，默认 ``RetryPolicy(max_retries=0)``。
    """

    def __init__(
        self,
        runtime: AgentRuntimeInterface,
        retry_policy: Optional[RetryPolicy] = None,
    ):
        self.runtime = runtime
        self.retry_policy = retry_policy or RetryPolicy(max_retries=0)

    def execute(self, context: HarnessContext) -> HarnessResult:
        start = time.perf_counter()
        retry_count = 0
        last_error: Optional[str] = None
        runtime_result: Optional[RuntimeResult] = None

        while True:
            runtime_result = self.runtime.run(context.to_runtime_input())
            if runtime_result.success:
                break

            last_error = runtime_result.error
            if not self.retry_policy.should_retry(last_error, retry_count):
                break
            retry_count += 1
            self.retry_policy.record(retry_count, last_error)

        if runtime_result is not None:
            context.update_from_runtime_result(runtime_result)

        elapsed = time.perf_counter() - start
        success = bool(runtime_result and runtime_result.success)
        metadata = ExecutionMetadata(
            request_id=context.request_id,
            session_id=context.session_id,
            status="success" if success else "failed",
            retry_count=retry_count,
            elapsed=round(elapsed, 6),
            error=runtime_result.error if runtime_result else last_error,
        )
        logger.info(f"execution metadata: {dict(metadata)}")

        return HarnessResult(
            success=success,
            output=runtime_result.output if runtime_result else None,
            context=context,
            runtime_result=runtime_result,
            retry_count=retry_count,
            error=runtime_result.error if runtime_result else last_error,
            metadata=dict(metadata),
        )
