"""ChatChat LangGraph Runtime 统一运行层（M06）。

提供：
- ``state``：``AgentState`` 统一状态抽象
- ``node``：Node 抽象与内建节点（Input/Passthrough/Output/RAG/LLM/Function）
- ``graph``：``GraphBuilder``（Sequential / Conditional / Loop）
- ``runtime``：``AgentRuntimeInterface`` + ``LangGraphRuntime``
- ``result``：``RuntimeResult``
- ``exceptions``：``GraphRuntimeError``（M01 体系）

M06 明确不做：Agent Harness / Registry / 业务 Agent / 复杂 Memory /
Retry / Guardrail / Streaming 重写 / Checkpoint 实现。
"""

from chatchat.agent.runtime.exceptions import GraphRuntimeError
from chatchat.agent.runtime.graph import (
    GraphBuilder,
    build_rag_chat_graph,
    max_iterations_condition,
)
from chatchat.agent.runtime.node import (
    FunctionNode,
    InputNode,
    LLMNode,
    NodeBase,
    OutputNode,
    PassthroughNode,
    RAGNode,
)
from chatchat.agent.runtime.result import RuntimeResult
from chatchat.agent.runtime.runtime import AgentRuntimeInterface, LangGraphRuntime
from chatchat.agent.runtime.state import AgentState

__all__ = [
    "AgentState",
    "NodeBase",
    "InputNode",
    "PassthroughNode",
    "OutputNode",
    "RAGNode",
    "LLMNode",
    "FunctionNode",
    "GraphBuilder",
    "build_rag_chat_graph",
    "max_iterations_condition",
    "AgentRuntimeInterface",
    "LangGraphRuntime",
    "RuntimeResult",
    "GraphRuntimeError",
]
