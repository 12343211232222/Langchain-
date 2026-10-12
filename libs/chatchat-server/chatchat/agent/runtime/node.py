"""Node 抽象与内建节点。

每个节点是一个可调用对象：``node(state) -> partial state dict``，
返回的部分 dict 会被 LangGraph 合并进 ``AgentState``。

内建节点：
- ``InputNode``：入口，把输入写入 state
- ``PassthroughNode``：透传，可用于占位/连接
- ``OutputNode``：收尾，写 execution_status=success
- ``RAGNode``：调用 M02 ``RAGServiceInterface.retrieve``
- ``LLMNode``：复用现有 LLM 调用层（``get_ChatOpenAI``），支持注入 mock
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from chatchat.core.logging import get_logger
from chatchat.agent.runtime.state import AgentState

logger = get_logger(__name__)


class NodeBase(ABC):
    """图节点抽象。子类实现 ``run``。"""

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__

    def __call__(self, state: AgentState) -> dict:
        logger.info(f"[node:{self.name}] start")
        update = self.run(state)
        logger.info(f"[node:{self.name}] done")
        return update or {}

    @abstractmethod
    def run(self, state: AgentState) -> dict:
        """执行节点逻辑，返回需要合并进 State 的部分 dict。"""
        raise NotImplementedError


class InputNode(NodeBase):
    """入口节点：把 ``user_input`` 写入状态并标记 running。"""

    def run(self, state: AgentState) -> dict:
        return {
            "user_input": state.get("user_input", ""),
            "execution_status": "running",
        }


class PassthroughNode(NodeBase):
    """透传节点：不修改状态（可选写入 context）。"""

    def __init__(self, name: Optional[str] = None, context_update: Optional[dict] = None):
        super().__init__(name)
        self.context_update = context_update or {}

    def run(self, state: AgentState) -> dict:
        if not self.context_update:
            return {}
        context = dict(state.get("context") or {})
        context.update(self.context_update)
        return {"context": context}


class OutputNode(NodeBase):
    """收尾节点：标记 success；若 final_output 为空则回退为 user_input。"""

    def run(self, state: AgentState) -> dict:
        return {
            "final_output": state.get("final_output") or state.get("user_input", ""),
            "execution_status": "success",
        }


class RAGNode(NodeBase):
    """RAG 检索节点：调用 M02 ``RAGServiceInterface``。

    ``rag_service`` 可注入（便于测试 / 替换实现）；为 None 时延迟使用
    ``chatchat.rag.get_rag_service()``。
    """

    def __init__(
        self,
        rag_service=None,
        knowledge_base: str = "",
        top_k: int = 5,
        retrieval_strategy: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self._rag_service = rag_service
        self.knowledge_base = knowledge_base
        self.top_k = top_k
        self.retrieval_strategy = retrieval_strategy

    def _service(self):
        if self._rag_service is None:
            # 延迟 import，避免循环依赖与不必要的重资源加载
            from chatchat.rag import get_rag_service

            self._rag_service = get_rag_service()
        return self._rag_service

    def run(self, state: AgentState) -> dict:
        from chatchat.rag.schemas import RAGQuery

        query = RAGQuery(
            query=state.get("user_input", ""),
            knowledge_base=self.knowledge_base,
            top_k=self.top_k,
            retrieval_strategy=self.retrieval_strategy,
        )
        response = self._service().retrieve(query)
        return {"retrieved_documents": list(response.documents)}


class LLMNode(NodeBase):
    """LLM 生成节点：复用现有 LLM 调用层。

    - ``llm`` 可注入（测试用 Fake/Mock）；为 None 时延迟调用
      ``chatchat.server.utils.get_ChatOpenAI``。
    - ``prompt_template`` 支持 ``{user_input}`` 与 ``{context}`` 两个占位符；
      为 None 时直接把 user_input 作为用户消息。
    - ``context`` 由 ``retrieved_documents`` 的内容拼接而成。
    """

    def __init__(
        self,
        llm=None,
        prompt_template: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self._llm = llm
        self.prompt_template = prompt_template

    def _model(self):
        if self._llm is None:
            # 延迟 import：复用现有 LLM 创建逻辑（配置驱动）
            from chatchat.server.utils import get_ChatOpenAI

            self._llm = get_ChatOpenAI(streaming=False)
        return self._llm

    def _render_prompt(self, state: AgentState) -> str:
        user_input = state.get("user_input", "")
        docs = state.get("retrieved_documents") or []
        context = "\n".join(getattr(d, "content", str(d)) for d in docs)
        if self.prompt_template is None:
            return user_input
        return self.prompt_template.format(user_input=user_input, context=context)

    def run(self, state: AgentState) -> dict:
        from langchain_core.messages import HumanMessage

        prompt = self._render_prompt(state)
        message = self._model().invoke([HumanMessage(content=prompt)])
        return {"final_output": getattr(message, "content", str(message))}


class FunctionNode(NodeBase):
    """把普通函数包装成节点，便于条件/Loop 中更新 context 等。"""

    def __init__(self, fn: Callable[[AgentState], dict], name: Optional[str] = None):
        super().__init__(name or getattr(fn, "__name__", None))
        self.fn = fn

    def run(self, state: AgentState) -> dict:
        return self.fn(state) or {}
