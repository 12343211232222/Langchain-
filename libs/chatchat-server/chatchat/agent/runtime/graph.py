"""Graph Builder：显式构建 LangGraph StateGraph。

支持三类 Edge：
- Sequential：``add_edge(src, dst)``
- Conditional：``add_conditional_edges(src, condition, mapping)``
- Loop：用 conditional edge 指回前驱节点 + 终止条件即成环
  （配合 ``max_iterations`` 辅助条件使用）

Builder 只做显式组装，不做任何自动规划。
"""

from typing import Any, Callable, Dict, Optional

from langgraph.graph import END, StateGraph

from chatchat.core.logging import get_logger
from chatchat.agent.runtime.exceptions import GraphRuntimeError
from chatchat.agent.runtime.node import NodeBase
from chatchat.agent.runtime.state import AgentState

logger = get_logger(__name__)


class GraphBuilder:
    """显式构建并编译 StateGraph。

    参数:
        state_schema: 状态 TypedDict 类，默认 ``AgentState``。
            业务 Agent 可传扩展 State（如 ``TenderReviewState``），
            消除「GraphBuilder vs 裸 StateGraph」两套构图风格。
    """

    def __init__(self, state_schema=AgentState):
        self._graph = StateGraph(state_schema)
        self._nodes: Dict[str, NodeBase] = {}
        self._outgoing: set = set()
        self._entry: Optional[str] = None
        self._built = False

    def add_node(self, name: str, node: NodeBase) -> "GraphBuilder":
        if self._built:
            raise GraphRuntimeError("图已编译，不能再添加节点")
        if name in self._nodes:
            raise GraphRuntimeError(f"节点重名: {name}")
        self._nodes[name] = node
        self._graph.add_node(name, node)
        return self

    def set_entry(self, name: str) -> "GraphBuilder":
        self._require_node(name)
        self._entry = name
        return self

    def add_edge(self, src: str, dst: str) -> "GraphBuilder":
        """Sequential 边。``dst`` 可为 ``END`` 表示流程结束。"""
        self._require_node(src)
        if dst != END:
            self._require_node(dst)
        self._graph.add_edge(src, dst)
        self._outgoing.add(src)
        return self

    def add_conditional_edges(
        self,
        src: str,
        condition: Callable[[AgentState], str],
        mapping: Dict[str, str],
    ) -> "GraphBuilder":
        """Conditional 边。

        ``condition(state)`` 返回 key，经 ``mapping`` 映射到目标节点名；
        目标可为 ``END``（``langgraph.graph.END``）表示结束。
        """
        self._require_node(src)
        for target in mapping.values():
            if target != END:
                self._require_node(target)
        self._graph.add_conditional_edges(src, condition, mapping)
        self._outgoing.add(src)
        return self

    def build(self):
        """编译并返回可调用的 LangGraph Runnable。

        自动为无出边的叶子节点追加 ``END`` 边，避免 dead-end 报错。
        """
        if self._built:
            raise GraphRuntimeError("图已编译，不能重复 build")
        if self._entry is None:
            raise GraphRuntimeError("未设置入口节点（set_entry）")
        if not self._nodes:
            raise GraphRuntimeError("图为空，至少需要一个节点")
        for name in self._nodes:
            if name not in self._outgoing:
                self._graph.add_edge(name, END)
        self._graph.set_entry_point(self._entry)
        compiled = self._graph.compile()
        self._built = True
        logger.info(f"graph compiled: nodes={list(self._nodes)} entry={self._entry}")
        return compiled

    def _require_node(self, name: str) -> None:
        if name not in self._nodes:
            raise GraphRuntimeError(f"节点不存在: {name}")


def max_iterations_condition(
    max_iterations: int,
    counter_key: str = "iteration",
) -> Callable[[AgentState], str]:
    """生成最小 Loop 终止条件。

    返回 ``"continue"`` / ``"exit"``：``context[counter_key] < max_iterations``
    时继续，否则退出。配合 conditional edge 指回前驱节点即成环。

    示例::

        def tick(state):
            ctx = dict(state.get("context") or {})
            ctx["iteration"] = ctx.get("iteration", 0) + 1
            return {"context": ctx}

        (GraphBuilder()
            .add_node("tick", FunctionNode(tick))
            .set_entry("tick")
            .add_conditional_edges("tick", max_iterations_condition(3),
                                   {"continue": "tick", "exit": END})
            .build())
    """

    def condition(state: AgentState) -> str:
        iteration = (state.get("context") or {}).get(counter_key, 0)
        return "continue" if iteration < max_iterations else "exit"

    return condition


def build_rag_chat_graph(
    rag_service=None,
    llm=None,
    knowledge_base: str = "",
    top_k: int = 5,
    retrieval_strategy: Optional[str] = None,
    prompt_template: Optional[str] = None,
):
    """最小 Demo 图：Input → RAG → LLM → Output。

    ``rag_service`` / ``llm`` 均可注入（为 None 时各节点使用默认实现）。
    """
    from chatchat.agent.runtime.node import InputNode, LLMNode, OutputNode, RAGNode

    return (
        GraphBuilder()
        .add_node("input", InputNode())
        .add_node(
            "rag",
            RAGNode(
                rag_service=rag_service,
                knowledge_base=knowledge_base,
                top_k=top_k,
                retrieval_strategy=retrieval_strategy,
            ),
        )
        .add_node("llm", LLMNode(llm=llm, prompt_template=prompt_template))
        .add_node("output", OutputNode())
        .set_entry("input")
        .add_edge("input", "rag")
        .add_edge("rag", "llm")
        .add_edge("llm", "output")
        .build()
    )
