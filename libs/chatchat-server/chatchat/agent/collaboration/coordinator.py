"""AgentCoordinator：多 Agent 协作编排器（M10）。

职责：接收 Plan → 通过 M08 Registry 发现并创建 Agent → 构建
AgentMessage → 调用 Agent.execute → 保存 AgentResult →
更新 CollaborationContext → 按依赖继续 → 返回 CollaborationResult。

边界：
- 不直接调用 LangGraphRuntime / Node
- 不 import 具体业务 Agent 类（只依赖 AgentRegistryInterface）
- Context / Memory / Policy / Retry 仍由 M07 Harness 负责，不复制到这里
- 单个 Agent 失败不抛出未捕获异常；依赖失败时下游 SKIPPED
"""

import json
from typing import Optional

from chatchat.agent.collaboration.context import CollaborationContext
from chatchat.agent.collaboration.exceptions import CollaborationError
from chatchat.agent.collaboration.schemas import (
    AgentMessage,
    AgentResult,
    CollaborationPlan,
    CollaborationResult,
    MessageType,
    TaskStatus,
)
from chatchat.agent.collaboration.strategies import (
    CollaborationStrategy,
    SequentialStrategy,
)
from chatchat.agent.registry import AgentRegistryInterface
from chatchat.core.logging import get_logger

logger = get_logger(__name__)


class AgentCoordinator:
    """Multi-Agent 协作编排器。"""

    def __init__(
        self,
        registry: AgentRegistryInterface,
        strategy: Optional[CollaborationStrategy] = None,
    ):
        self.registry = registry
        self.strategy = strategy or SequentialStrategy()

    def execute(
        self,
        plan: CollaborationPlan,
        context: Optional[CollaborationContext] = None,
    ) -> CollaborationResult:
        context = context or CollaborationContext()
        result = CollaborationResult(success=True, plan_id=plan.plan_id, context=context)

        try:
            ordered = self.strategy.order(plan)
        except CollaborationError as e:
            result.success = False
            result.error = str(e)
            return result

        for task in ordered:
            # 依赖检查：任一上游失败/跳过 → 本任务跳过
            failed_deps = [
                d for d in task.dependencies
                if not context.task_results.get(d)
                or not context.task_results[d].success
            ]
            if failed_deps:
                task.status = TaskStatus.SKIPPED.value
                skipped = AgentResult(
                    task_id=task.task_id,
                    agent_id=task.agent_id,
                    success=False,
                    error=f"dependency failure: {failed_deps}",
                )
                context.add_result(skipped)
                result.results[task.task_id] = skipped
                result.success = False
                result.failed_task = result.failed_task or failed_deps[0]
                result.error = result.error or skipped.error
                logger.info(f"task skipped (dependency failure): {task.task_id}")
                continue

            task.status = TaskStatus.RUNNING.value
            agent_result = self._execute_task(task, plan, context)
            context.add_result(agent_result)
            result.results[task.task_id] = agent_result
            task.status = (
                TaskStatus.SUCCESS.value if agent_result.success
                else TaskStatus.FAILED.value
            )
            if not agent_result.success:
                result.success = False
                result.failed_task = result.failed_task or task.task_id
                result.error = result.error or agent_result.error

        logger.info(
            f"collaboration finished: plan={plan.plan_id} "
            f"success={result.success}"
        )
        return result

    def _execute_task(self, task, plan, context) -> AgentResult:
        """执行单个任务：发现 Agent → 发消息 → 执行 → 结构化结果。"""
        # 1. 发现 Agent（M08 Registry）
        if not self.registry.exists(task.agent_id):
            error = f"agent 不存在: {task.agent_id}"
            self._record_error(context, task, error)
            return AgentResult(task_id=task.task_id, agent_id=task.agent_id,
                               success=False, error=error)
        # 2. 创建 Agent
        try:
            agent = self.registry.create(task.agent_id)
        except Exception as e:
            error = f"agent 创建失败({task.agent_id}): {e}"
            self._record_error(context, task, error)
            return AgentResult(task_id=task.task_id, agent_id=task.agent_id,
                               success=False, error=error)
        # 3. 构建结构化 AgentMessage（只携带本任务所需数据）
        payload = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "input": task.input if task.input is not None else context.user_input,
            "dependencies": context.get_dependency_outputs(task.dependencies),
        }
        request_message = AgentMessage(
            sender="coordinator",
            receiver=task.agent_id,
            message_type=MessageType.TASK_REQUEST.value,
            content=payload,
            correlation_id=context.collaboration_id,
        )
        context.add_message(request_message)
        # 4. 执行（Agent → Harness → Runtime，由 Agent 内部负责）
        try:
            harness_result = agent.execute(json.dumps(payload, ensure_ascii=False))
        except Exception as e:
            error = f"agent 执行异常({task.agent_id}): {e}"
            self._record_error(context, task, error)
            return AgentResult(task_id=task.task_id, agent_id=task.agent_id,
                               success=False, error=error)
        # 5. 结构化结果 + TASK_RESULT 消息
        agent_result = AgentResult(
            task_id=task.task_id,
            agent_id=task.agent_id,
            success=harness_result.success,
            output=harness_result.output,
            error=harness_result.error,
            metadata=dict(harness_result.metadata),
        )
        context.add_message(AgentMessage(
            sender=task.agent_id,
            receiver="coordinator",
            message_type=(
                MessageType.TASK_RESULT.value if agent_result.success
                else MessageType.ERROR.value
            ),
            content=agent_result.output if agent_result.success else agent_result.error,
            correlation_id=context.collaboration_id,
        ))
        return agent_result

    @staticmethod
    def _record_error(context, task, error: str) -> None:
        logger.error(error)
        context.add_message(AgentMessage(
            sender="coordinator",
            receiver=task.agent_id,
            message_type=MessageType.ERROR.value,
            content=error,
            correlation_id=context.collaboration_id,
        ))
