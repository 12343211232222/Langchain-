"""M06 LangGraph Runtime 统一运行层测试。"""

import pytest
from langgraph.graph import END

from chatchat.agent.runtime import (
    AgentRuntimeInterface,
    AgentState,
    FunctionNode,
    GraphBuilder,
    GraphRuntimeError,
    InputNode,
    LangGraphRuntime,
    LLMNode,
    OutputNode,
    PassthroughNode,
    RAGNode,
    RuntimeResult,
    build_rag_chat_graph,
    max_iterations_condition,
)
from chatchat.core.exceptions import ChatchatError, ServiceError
from chatchat.rag.schemas import RAGDocument, RAGResponse


# ---------------------------------------------------------------- helpers


class FakeMessage:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    """模拟 get_ChatOpenAI 返回的 ChatOpenAI：invoke(messages) -> message。"""

    def __init__(self, reply: str = "fake reply"):
        self.reply = reply
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[-1].content)
        return FakeMessage(self.reply)


class MockRAGService:
    """实现 RAGServiceInterface.retrieve 的 mock。"""

    def __init__(self, docs=None):
        self.docs = docs or [RAGDocument(content="知识片段A"), RAGDocument(content="知识片段B")]
        self.queries = []

    def retrieve(self, request):
        self.queries.append(request)
        return RAGResponse(query=request.query, documents=list(self.docs))


def tick(state):
    ctx = dict(state.get("context") or {})
    ctx["iteration"] = ctx.get("iteration", 0) + 1
    return {"context": ctx}


# ---------------------------------------------------------------- 1. State


class TestAgentState:
    def test_state_is_typed_dict_subset(self):
        state: AgentState = {"user_input": "hi"}
        assert state["user_input"] == "hi"
        assert "messages" not in state  # total=False：字段全部可选

    def test_all_declared_fields_assignable(self):
        state: AgentState = {
            "user_input": "q",
            "messages": [],
            "context": {},
            "retrieved_documents": [],
            "tool_results": [],
            "final_output": "a",
            "execution_status": "running",
            "error": "",
        }
        assert state["execution_status"] == "running"


# ---------------------------------------------------------------- 2. Node


class TestNodes:
    def test_input_node_marks_running(self):
        update = InputNode()({"user_input": "hello"})
        assert update["user_input"] == "hello"
        assert update["execution_status"] == "running"

    def test_passthrough_node_noop(self):
        assert PassthroughNode()({"user_input": "x"}) == {}

    def test_passthrough_node_context_update(self):
        update = PassthroughNode(context_update={"k": "v"})({"context": {"a": 1}})
        assert update["context"] == {"a": 1, "k": "v"}

    def test_output_node_success_and_fallback(self):
        update = OutputNode()({"user_input": "q", "final_output": "ans"})
        assert update == {"final_output": "ans", "execution_status": "success"}
        # final_output 为空时回退 user_input
        update = OutputNode()({"user_input": "q"})
        assert update["final_output"] == "q"

    def test_llm_node_with_mock_llm(self):
        llm = FakeLLM("回答：42")
        update = LLMNode(llm=llm)({"user_input": "问题"})
        assert update["final_output"] == "回答：42"
        assert llm.prompts == ["问题"]

    def test_llm_node_prompt_template_renders_context(self):
        llm = FakeLLM()
        node = LLMNode(llm=llm, prompt_template="资料:{context}\n问题:{user_input}")
        node({
            "user_input": "是什么",
            "retrieved_documents": [RAGDocument(content="片段X")],
        })
        assert llm.prompts == ["资料:片段X\n问题:是什么"]


# ---------------------------------------------------------------- 3. Sequential


class TestSequentialGraph:
    def test_sequential_execution_order(self):
        order = []

        def mark(tag):
            def fn(state):
                order.append(tag)
                return {}

            return fn

        graph = (
            GraphBuilder()
            .add_node("a", FunctionNode(mark("a")))
            .add_node("b", FunctionNode(mark("b")))
            .add_node("c", FunctionNode(mark("c")))
            .set_entry("a")
            .add_edge("a", "b")
            .add_edge("b", "c")
            .build()
        )
        graph.invoke({"user_input": "x"})
        assert order == ["a", "b", "c"]

    def test_state_flows_through_sequential_nodes(self):
        graph = (
            GraphBuilder()
            .add_node("input", InputNode())
            .add_node("output", OutputNode())
            .set_entry("input")
            .add_edge("input", "output")
            .build()
        )
        final = graph.invoke({"user_input": "你好"})
        assert final["execution_status"] == "success"
        assert final["final_output"] == "你好"


# ---------------------------------------------------------------- 4. Conditional


class TestConditionalGraph:
    def _build(self):
        def route(state):
            return "yes" if "?" in state.get("user_input", "") else "no"

        def answer(state):
            return {"final_output": "解答"}

        def chat(state):
            return {"final_output": "闲聊"}

        return (
            GraphBuilder()
            .add_node("input", InputNode())
            .add_node("answer", FunctionNode(answer))
            .add_node("chat", FunctionNode(chat))
            .set_entry("input")
            .add_conditional_edges("input", route, {"yes": "answer", "no": "chat"})
            .build()
        )

    def test_conditional_routes_to_matching_branch(self):
        graph = self._build()
        assert graph.invoke({"user_input": "1+1?"})["final_output"] == "解答"
        assert graph.invoke({"user_input": "你好"})["final_output"] == "闲聊"


# ---------------------------------------------------------------- 5. Loop


class TestLoopGraph:
    def test_loop_terminates_at_max_iterations(self):
        graph = (
            GraphBuilder()
            .add_node("tick", FunctionNode(tick))
            .set_entry("tick")
            .add_conditional_edges(
                "tick",
                max_iterations_condition(3),
                {"continue": "tick", "exit": END},
            )
            .build()
        )
        final = graph.invoke({"user_input": "x"})
        assert final["context"]["iteration"] == 3

    def test_runtime_recursion_limit_guards_runaway_loop(self):
        # 永不满足的退出条件：应由 recursion_limit 兜底
        graph = (
            GraphBuilder()
            .add_node("tick", FunctionNode(tick))
            .set_entry("tick")
            .add_conditional_edges(
                "tick", lambda s: "continue", {"continue": "tick", "exit": END}
            )
            .build()
        )
        result = LangGraphRuntime(graph, recursion_limit=10).run("x")
        assert result.success is False
        assert result.error is not None


# ---------------------------------------------------------------- 6. RAGNode


class TestRAGNode:
    def test_rag_node_calls_service_and_stores_documents(self):
        service = MockRAGService()
        node = RAGNode(rag_service=service, knowledge_base="kb1", top_k=3,
                       retrieval_strategy="hybrid_rrf")
        update = node({"user_input": "查询内容"})
        assert len(update["retrieved_documents"]) == 2
        query = service.queries[0]
        assert query.query == "查询内容"
        assert query.knowledge_base == "kb1"
        assert query.top_k == 3
        assert query.retrieval_strategy == "hybrid_rrf"

    def test_rag_chat_demo_graph(self):
        service = MockRAGService()
        llm = FakeLLM("基于知识的回答")
        graph = build_rag_chat_graph(
            rag_service=service,
            llm=llm,
            knowledge_base="kb1",
            prompt_template="资料:{context}\n问题:{user_input}",
        )
        final = graph.invoke({"user_input": "什么是RAG"})
        assert final["execution_status"] == "success"
        assert final["final_output"] == "基于知识的回答"
        assert "知识片段A" in llm.prompts[0]


# ---------------------------------------------------------------- 7. Runtime


class TestLangGraphRuntime:
    def _simple_graph(self):
        return (
            GraphBuilder()
            .add_node("input", InputNode())
            .add_node("output", OutputNode())
            .set_entry("input")
            .add_edge("input", "output")
            .build()
        )

    def test_implements_interface(self):
        assert isinstance(LangGraphRuntime(self._simple_graph()), AgentRuntimeInterface)

    def test_run_with_string_input(self):
        result = LangGraphRuntime(self._simple_graph()).run("hello")
        assert isinstance(result, RuntimeResult)
        assert result.success is True
        assert result.output == "hello"
        assert result.error is None
        assert result.state["execution_status"] == "success"

    def test_run_with_dict_input(self):
        result = LangGraphRuntime(self._simple_graph()).run(
            {"user_input": "hi", "context": {"session": "s1"}}
        )
        assert result.success is True
        assert result.state["context"]["session"] == "s1"


# ---------------------------------------------------------------- 8. Error


class TestErrorHandling:
    def test_node_chatchat_error_becomes_failed_result(self):
        def boom(state):
            raise ServiceError("检索服务不可用")

        graph = (
            GraphBuilder()
            .add_node("bad", FunctionNode(boom))
            .set_entry("bad")
            .build()
        )
        result = LangGraphRuntime(graph).run("x")
        assert result.success is False
        assert "检索服务不可用" in result.error
        assert result.state["execution_status"] == "failed"

    def test_node_unexpected_error_becomes_failed_result(self):
        def boom(state):
            raise ValueError("unexpected")

        graph = (
            GraphBuilder()
            .add_node("bad", FunctionNode(boom))
            .set_entry("bad")
            .build()
        )
        result = LangGraphRuntime(graph).run("x")
        assert result.success is False
        assert "ValueError" in result.error

    def test_graph_builder_validation_errors(self):
        builder = GraphBuilder()
        with pytest.raises(GraphRuntimeError):
            builder.build()  # 无入口
        with pytest.raises(GraphRuntimeError):
            builder.add_node("n", PassthroughNode()).set_entry("n").add_edge("n", "missing")
        with pytest.raises(GraphRuntimeError):
            GraphBuilder().add_node("n", PassthroughNode()).build()  # 缺入口

    def test_graph_runtime_error_is_chatchat_error(self):
        assert issubclass(GraphRuntimeError, ChatchatError)
