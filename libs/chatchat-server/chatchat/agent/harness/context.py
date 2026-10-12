"""Context Manager：统一 Harness 上下文抽象。

``HarnessContext`` 是 Harness 层的上下文载体，通过 ``to_runtime_input`` /
``update_from_runtime_result`` 与 M06 ``AgentState`` 双向适配，
不复制一套独立状态。

ContextManager 只负责：初始化 / 合并 / 更新 / 读取 / 生命周期管理，
不引入 Context Database。
"""

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chatchat.agent.runtime.result import RuntimeResult
from chatchat.agent.runtime.state import AgentState


@dataclass
class HarnessRequest:
    """一次 Harness 执行的输入。"""

    user_input: str
    session_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HarnessContext:
    """Harness 层统一上下文。

    字段:
        request_id: 本次执行唯一标识（自动生成）。
        session_id: 会话标识（跨多次执行，由调用方提供）。
        user_input: 用户原始输入。
        metadata: 调用方附加信息。
        runtime_context: 最近一次 Runtime 返回的 AgentState 快照。
        retrieved_documents: 最近一次 RAGNode 检索结果。
        tool_results: 工具执行结果（M07 预留）。
        variables: 自由变量（跨节点/跨执行传递）。
    """

    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    session_id: str = ""
    user_input: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    runtime_context: Dict[str, Any] = field(default_factory=dict)
    retrieved_documents: List[Any] = field(default_factory=list)
    tool_results: List[Any] = field(default_factory=list)
    variables: Dict[str, Any] = field(default_factory=dict)

    def to_runtime_input(self) -> AgentState:
        """适配为 M06 ``AgentRuntimeInterface.run`` 的输入。"""
        return {
            "user_input": self.user_input,
            "context": {
                "request_id": self.request_id,
                "session_id": self.session_id,
                **self.variables,
            },
        }

    def update_from_runtime_result(self, result: RuntimeResult) -> None:
        """用 M06 RuntimeResult 回写上下文。"""
        state = result.state or {}
        self.runtime_context = dict(state)
        if state.get("retrieved_documents"):
            self.retrieved_documents = list(state["retrieved_documents"])
        if state.get("tool_results"):
            self.tool_results = list(state["tool_results"])


class ContextManager(ABC):
    """Context 生命周期管理接口。"""

    @abstractmethod
    def create(self, request: HarnessRequest) -> HarnessContext:
        """按请求初始化上下文。"""
        raise NotImplementedError

    @abstractmethod
    def update(self, context: HarnessContext, **updates: Any) -> HarnessContext:
        """合并更新上下文字段。"""
        raise NotImplementedError

    @abstractmethod
    def get(self, request_id: str) -> Optional[HarnessContext]:
        """按 request_id 读取上下文。"""
        raise NotImplementedError

    @abstractmethod
    def clear(self, request_id: str) -> None:
        """生命周期结束，清理上下文。"""
        raise NotImplementedError


class DefaultContextManager(ContextManager):
    """内存版 ContextManager（单进程、无外部依赖）。"""

    def __init__(self):
        self._contexts: Dict[str, HarnessContext] = {}

    def create(self, request: HarnessRequest) -> HarnessContext:
        context = HarnessContext(
            session_id=request.session_id,
            user_input=request.user_input,
            metadata=dict(request.metadata),
        )
        self._contexts[context.request_id] = context
        return context

    def update(self, context: HarnessContext, **updates: Any) -> HarnessContext:
        for key, value in updates.items():
            if not hasattr(context, key):
                raise KeyError(f"HarnessContext 无字段: {key}")
            setattr(context, key, value)
        return context

    def get(self, request_id: str) -> Optional[HarnessContext]:
        return self._contexts.get(request_id)

    def clear(self, request_id: str) -> None:
        self._contexts.pop(request_id, None)
