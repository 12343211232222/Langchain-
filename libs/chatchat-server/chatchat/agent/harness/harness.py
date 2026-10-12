"""DefaultAgentHarness：M07 统一执行入口。

编排顺序（见 M07 规格）：
request -> Context 初始化 -> Policy Before -> Execution Manager(M06 Runtime)
-> Context/Memory 更新 -> Policy After -> HarnessResult

Harness 不直接执行 LangGraph Node，只调用 M06 ``AgentRuntimeInterface``。
"""

from typing import Optional, Union

from chatchat.agent.harness.context import (
    ContextManager,
    DefaultContextManager,
    HarnessRequest,
)
from chatchat.agent.harness.execution import ExecutionManager
from chatchat.agent.harness.interfaces import AgentHarnessInterface
from chatchat.agent.harness.memory import InMemoryMemoryManager, MemoryManager
from chatchat.agent.harness.policy import AllowAllPolicy, ExecutionPolicy
from chatchat.agent.harness.result import HarnessResult
from chatchat.agent.harness.retry import RetryPolicy
from chatchat.agent.runtime import AgentRuntimeInterface
from chatchat.core.logging import get_logger

logger = get_logger(__name__)


class DefaultAgentHarness(AgentHarnessInterface):
    """默认 Agent Harness 实现。

    参数:
        runtime: M06 ``AgentRuntimeInterface``。
        context_manager: 默认 ``DefaultContextManager``。
        memory_manager: 默认 ``InMemoryMemoryManager``。
        policy: 默认 ``AllowAllPolicy``。
        retry_policy: 默认 ``RetryPolicy(max_retries=0)``。
    """

    def __init__(
        self,
        runtime: AgentRuntimeInterface,
        context_manager: Optional[ContextManager] = None,
        memory_manager: Optional[MemoryManager] = None,
        policy: Optional[ExecutionPolicy] = None,
        retry_policy: Optional[RetryPolicy] = None,
    ):
        self.runtime = runtime
        self.context_manager = context_manager or DefaultContextManager()
        self.memory_manager = memory_manager or InMemoryMemoryManager()
        self.policy = policy or AllowAllPolicy()
        self.execution_manager = ExecutionManager(
            runtime=runtime,
            retry_policy=retry_policy or RetryPolicy(max_retries=0),
        )

    def execute(self, request: Union[HarnessRequest, str]) -> HarnessResult:
        if isinstance(request, str):
            request = HarnessRequest(user_input=request)

        context = self.context_manager.create(request)
        logger.info(
            f"harness execute: request_id={context.request_id} "
            f"session_id={context.session_id}"
        )

        self.policy.before_execute(context)

        result = self.execution_manager.execute(context)

        if not result.success and result.error:
            self.policy.on_error(context, result.error)

        self.memory_manager.save(
            request.session_id or context.request_id,
            {
                "request_id": context.request_id,
                "user_input": request.user_input,
                "output": result.output,
                "success": result.success,
                "retry_count": result.retry_count,
            },
        )

        if result.success and result.runtime_result is not None:
            self.policy.after_execute(context, result.runtime_result)

        return result
