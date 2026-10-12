"""Agent Runtime 接口与 LangGraph 实现。

``LangGraphRuntime.run`` 是同步最小实现：
- 输入 ``input_data``（str 或 dict）→ 构建初始 ``AgentState``
- 调用编译后的图；捕获一切异常并转换为 ``RuntimeResult(success=False)``
- 不重写 Streaming（M06 明确排除），Checkpoint 仅预留字段

异步路径（审计修复）：``arun`` 基于 ``graph.ainvoke``，供 FastAPI/SSE
等异步环境非阻塞调用；``astream`` 基于 ``graph.astream`` 逐步产出
各节点状态快照。两者语义与 ``run`` 一致（统一 RuntimeResult / 异常不吞）。
"""

import time
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, Optional, Union

from chatchat.core.logging import get_logger
from chatchat.agent.runtime.result import RuntimeResult
from chatchat.agent.runtime.state import AgentState

logger = get_logger(__name__)

# langgraph 默认 recursion_limit=25；作为最小 Loop 的保护上限
DEFAULT_RECURSION_LIMIT = 50


class AgentRuntimeInterface(ABC):
    """Agent Runtime 抽象接口。"""

    @abstractmethod
    def run(self, input_data: Union[str, Dict[str, Any]]) -> RuntimeResult:
        """执行一次图运行，返回统一 ``RuntimeResult``。"""
        raise NotImplementedError

    async def arun(self, input_data: Union[str, Dict[str, Any]]) -> RuntimeResult:
        """异步执行（默认回退到同步实现；LangGraphRuntime 覆写为 ainvoke）。"""
        return self.run(input_data)


class LangGraphRuntime(AgentRuntimeInterface):
    """基于已编译 LangGraph 的 Runtime。

    参数:
        graph: ``GraphBuilder.build()`` 的返回（LangGraph Runnable）。
        recursion_limit: 图执行步数上限（langgraph ``recursion_limit``），
            防止 Loop 无终止。None 使用默认值。
    """

    def __init__(self, graph, recursion_limit: Optional[int] = None):
        self.graph = graph
        self.recursion_limit = recursion_limit or DEFAULT_RECURSION_LIMIT

    def run(self, input_data: Union[str, Dict[str, Any]]) -> RuntimeResult:
        start = time.perf_counter()
        state = self._normalize(input_data)
        try:
            final_state: AgentState = self.graph.invoke(
                state,
                config={"recursion_limit": self.recursion_limit},
            )
        except Exception as e:
            return self._failure(state, e, start)
        return self._success(final_state, start)

    async def arun(self, input_data: Union[str, Dict[str, Any]]) -> RuntimeResult:
        """异步执行（graph.ainvoke），语义与 ``run`` 一致。"""
        start = time.perf_counter()
        state = self._normalize(input_data)
        try:
            final_state: AgentState = await self.graph.ainvoke(
                state,
                config={"recursion_limit": self.recursion_limit},
            )
        except Exception as e:
            return self._failure(state, e, start)
        return self._success(final_state, start)

    async def astream(
        self, input_data: Union[str, Dict[str, Any]]
    ) -> AsyncIterator[Dict[str, Any]]:
        """异步流式执行：逐步产出 ``{node_name: state_update}`` 快照。"""
        state = self._normalize(input_data)
        try:
            async for event in self.graph.astream(
                state,
                config={"recursion_limit": self.recursion_limit},
            ):
                yield event
        except Exception as e:
            logger.error(f"runtime astream failed: {type(e).__name__}: {e}")
            yield {"error": f"{type(e).__name__}: {e}"}

    @staticmethod
    def _normalize(input_data: Union[str, Dict[str, Any]]) -> AgentState:
        if isinstance(input_data, str):
            state: AgentState = {"user_input": input_data}
        else:
            state = dict(input_data)  # type: ignore[assignment]
        state.setdefault("execution_status", "pending")
        return state

    def _failure(self, state: AgentState, e: Exception, start: float) -> RuntimeResult:
        # 不吞异常：记录日志并转换为统一失败结果
        logger.error(f"runtime run failed: {type(e).__name__}: {e}")
        state["execution_status"] = "failed"
        state["error"] = f"{type(e).__name__}: {e}"
        return RuntimeResult(
            success=False,
            output=None,
            state=state,
            error=state["error"],
            metadata={"elapsed": time.perf_counter() - start},
        )

    @staticmethod
    def _success(final_state: AgentState, start: float) -> RuntimeResult:
        success = final_state.get("execution_status") != "failed"
        return RuntimeResult(
            success=success,
            output=final_state.get("final_output"),
            state=final_state,
            error=final_state.get("error"),
            metadata={"elapsed": time.perf_counter() - start},
        )
