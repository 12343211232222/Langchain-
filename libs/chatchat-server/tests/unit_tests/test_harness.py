"""M07 Agent Harness 统一运行控制层测试。

测试全部使用 Mock Runtime / FakeLLM，不调用真实 LLM。
"""

import pytest

from chatchat.agent.harness import (
    AgentHarnessInterface,
    AllowAllPolicy,
    DefaultAgentHarness,
    DefaultContextManager,
    ExecutionManager,
    ExecutionPolicy,
    HarnessContext,
    HarnessError,
    HarnessRequest,
    HarnessResult,
    InMemoryMemoryManager,
    PolicyViolationError,
    RegistryToolManager,
    RetryPolicy,
)
from chatchat.agent.runtime import (
    AgentRuntimeInterface,
    GraphBuilder,
    InputNode,
    LangGraphRuntime,
    OutputNode,
    RuntimeResult,
    build_rag_chat_graph,
)
from chatchat.core.exceptions import ChatchatError
from chatchat.rag.schemas import RAGDocument, RAGResponse


# ---------------------------------------------------------------- helpers


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, reply="fake answer"):
        self.reply = reply

    def invoke(self, messages):
        return FakeMessage(self.reply)


class MockRAGService:
    def retrieve(self, request):
        return RAGResponse(
            query=request.query,
            documents=[RAGDocument(content="知识片段A")],
        )


class MockRuntime(AgentRuntimeInterface):
    """可编程的 Mock Runtime：按队列依次返回结果。"""

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def run(self, input_data):
        self.calls += 1
        return self.results.pop(0)


def ok_result(output="done"):
    return RuntimeResult(
        success=True,
        output=output,
        state={"user_input": "q", "final_output": output, "execution_status": "success"},
    )


def fail_result(error="ServiceError: boom"):
    return RuntimeResult(
        success=False,
        output=None,
        state={"user_input": "q", "execution_status": "failed", "error": error},
        error=error,
    )


def simple_runtime():
    graph = (
        GraphBuilder()
        .add_node("input", InputNode())
        .add_node("output", OutputNode())
        .set_entry("input")
        .add_edge("input", "output")
        .build()
    )
    return LangGraphRuntime(graph)


# ---------------------------------------------------------------- 1. ContextManager


class TestContextManager:
    def test_create_assigns_ids_and_stores(self):
        cm = DefaultContextManager()
        ctx = cm.create(HarnessRequest(user_input="hi", session_id="s1"))
        assert ctx.request_id
        assert ctx.session_id == "s1"
        assert ctx.user_input == "hi"
        assert cm.get(ctx.request_id) is ctx

    def test_update_merges_fields(self):
        cm = DefaultContextManager()
        ctx = cm.create(HarnessRequest(user_input="a"))
        cm.update(ctx, variables={"k": "v"})
        assert ctx.variables == {"k": "v"}
        with pytest.raises(KeyError):
            cm.update(ctx, no_such_field=1)

    def test_clear_removes_context(self):
        cm = DefaultContextManager()
        ctx = cm.create(HarnessRequest(user_input="a"))
        cm.clear(ctx.request_id)
        assert cm.get(ctx.request_id) is None

    def test_context_adapts_to_agent_state(self):
        ctx = HarnessContext(session_id="s1", user_input="问题",
                             variables={"v": 1})
        state = ctx.to_runtime_input()
        assert state["user_input"] == "问题"
        assert state["context"]["session_id"] == "s1"
        assert state["context"]["v"] == 1

    def test_context_updates_from_runtime_result(self):
        ctx = HarnessContext()
        result = RuntimeResult(
            success=True,
            state={
                "retrieved_documents": [RAGDocument(content="d")],
                "final_output": "out",
            },
        )
        ctx.update_from_runtime_result(result)
        assert len(ctx.retrieved_documents) == 1
        assert ctx.runtime_context["final_output"] == "out"


# ---------------------------------------------------------------- 2. MemoryManager


class TestMemoryManager:
    def test_create_update_get(self):
        mm = InMemoryMemoryManager()
        mm.create("s1")
        mm.update("s1", {"turn": 1})
        assert mm.get("s1") == [{"turn": 1}]

    def test_save_load(self):
        mm = InMemoryMemoryManager()
        mm.save("s2", {"turn": "a"})
        assert mm.load("s2") == [{"turn": "a"}]

    def test_clear(self):
        mm = InMemoryMemoryManager()
        mm.update("s3", {"x": 1})
        mm.clear("s3")
        assert mm.get("s3") == []


# ---------------------------------------------------------------- 3. ToolManager


class TestToolManager:
    def test_register_get_list_execute(self):
        tm = RegistryToolManager()
        tm.register("echo", lambda text="": f"echo:{text}")
        assert tm.get("echo") is not None
        assert "echo" in tm.list_tools()
        assert tm.execute("echo", text="hi") == "echo:hi"

    def test_execute_missing_tool_raises(self):
        tm = RegistryToolManager()
        with pytest.raises(KeyError):
            tm.execute("no_such_tool_xyz")

    def test_clear_only_harness_side(self):
        tm = RegistryToolManager()
        tm.register("tmp", lambda: 1)
        tm.clear()
        assert tm.get("tmp") is None


# ---------------------------------------------------------------- 4. RetryPolicy


class TestRetryPolicy:
    def test_default_no_retry(self):
        policy = RetryPolicy()
        assert policy.max_retries == 0
        assert policy.should_retry("ServiceError: x", 0) is False

    def test_retryable_exception_match(self):
        policy = RetryPolicy(max_retries=2, retryable_exceptions={"ServiceError"})
        assert policy.should_retry("ServiceError: boom", 0) is True
        assert policy.should_retry("ServiceError: boom", 2) is False  # 上限

    def test_non_retryable_exception(self):
        policy = RetryPolicy(max_retries=2, retryable_exceptions={"ServiceError"})
        assert policy.should_retry("ValidationError: bad input", 0) is False
        assert policy.should_retry(None, 0) is False


# ---------------------------------------------------------------- 5. AllowAllPolicy


class TestAllowAllPolicy:
    def test_hooks_do_not_change_behavior(self):
        policy = AllowAllPolicy()
        ctx = HarnessContext()
        policy.before_execute(ctx)
        policy.after_execute(ctx, ok_result())
        policy.on_error(ctx, "err")
        assert isinstance(policy, ExecutionPolicy)


class RecordingPolicy(AllowAllPolicy):
    def __init__(self):
        self.calls = []

    def before_execute(self, context):
        self.calls.append("before")

    def after_execute(self, context, result):
        self.calls.append("after")

    def on_error(self, context, error):
        self.calls.append("on_error")


# ---------------------------------------------------------------- 6. ExecutionManager


class TestExecutionManager:
    def test_success_updates_context_and_metadata(self):
        em = ExecutionManager(MockRuntime([ok_result("ok")]))
        ctx = HarnessContext(user_input="q")
        result = em.execute(ctx)
        assert result.success is True
        assert result.output == "ok"
        assert result.metadata["status"] == "success"
        assert ctx.runtime_context["final_output"] == "ok"

    def test_failure_without_retry(self):
        em = ExecutionManager(MockRuntime([fail_result()]))
        result = em.execute(HarnessContext(user_input="q"))
        assert result.success is False
        assert "ServiceError" in result.error
        assert result.retry_count == 0


# ---------------------------------------------------------------- 7. HarnessResult


class TestHarnessResult:
    def test_fields(self):
        r = HarnessResult(success=True, output="o", retry_count=1)
        assert r.success and r.output == "o" and r.retry_count == 1
        assert r.context is None and r.runtime_result is None


# ---------------------------------------------------------------- 8. DefaultAgentHarness


class TestDefaultAgentHarness:
    def test_implements_interface(self):
        h = DefaultAgentHarness(MockRuntime([ok_result()]))
        assert isinstance(h, AgentHarnessInterface)

    def test_execute_full_chain_success(self):
        h = DefaultAgentHarness(MockRuntime([ok_result("最终答案")]))
        result = h.execute(HarnessRequest(user_input="问题", session_id="s1"))
        assert result.success is True
        assert result.output == "最终答案"
        assert result.context.user_input == "问题"
        assert result.metadata["request_id"] == result.context.request_id

    def test_execute_accepts_plain_string(self):
        h = DefaultAgentHarness(MockRuntime([ok_result()]))
        assert h.execute("你好").success is True


# ---------------------------------------------------------------- 9. Runtime 成功


class TestRuntimeSuccessPath:
    def test_harness_over_real_langgraph_runtime(self):
        h = DefaultAgentHarness(simple_runtime())
        result = h.execute("hello")
        assert result.success is True
        assert result.output == "hello"
        assert result.context.runtime_context["execution_status"] == "success"


# ---------------------------------------------------------------- 10. Runtime 失败


class TestRuntimeFailurePath:
    def test_failure_propagates_to_harness_result(self):
        h = DefaultAgentHarness(MockRuntime([fail_result()]))
        result = h.execute("x")
        assert result.success is False
        assert result.error and "ServiceError" in result.error
        assert result.output is None


# ---------------------------------------------------------------- 11. Retry


class TestRetry:
    def test_retry_until_success(self):
        runtime = MockRuntime([fail_result(), ok_result("重试成功")])
        h = DefaultAgentHarness(
            runtime,
            retry_policy=RetryPolicy(max_retries=2,
                                     retryable_exceptions={"ServiceError"}),
        )
        result = h.execute("x")
        assert result.success is True
        assert result.output == "重试成功"
        assert result.retry_count == 1
        assert runtime.calls == 2

    def test_retry_exhausted_returns_failure(self):
        runtime = MockRuntime([fail_result(), fail_result(), fail_result()])
        h = DefaultAgentHarness(
            runtime,
            retry_policy=RetryPolicy(max_retries=2,
                                     retryable_exceptions={"ServiceError"}),
        )
        result = h.execute("x")
        assert result.success is False
        assert result.retry_count == 2
        assert runtime.calls == 3


# ---------------------------------------------------------------- 12. 不可重试异常


class TestNonRetryable:
    def test_non_retryable_error_not_retried(self):
        runtime = MockRuntime([fail_result("ValidationError: bad")])
        h = DefaultAgentHarness(
            runtime,
            retry_policy=RetryPolicy(max_retries=3,
                                     retryable_exceptions={"ServiceError"}),
        )
        result = h.execute("x")
        assert result.success is False
        assert result.retry_count == 0
        assert runtime.calls == 1


# ---------------------------------------------------------------- 13. Policy Hook


class TestPolicyHooks:
    def test_hooks_called_in_order(self):
        policy = RecordingPolicy()
        h = DefaultAgentHarness(MockRuntime([ok_result()]), policy=policy)
        h.execute("x")
        assert policy.calls == ["before", "after"]

    def test_on_error_called_on_failure(self):
        policy = RecordingPolicy()
        h = DefaultAgentHarness(MockRuntime([fail_result()]), policy=policy)
        h.execute("x")
        assert policy.calls == ["before", "on_error"]

    def test_policy_can_block_execution(self):
        class BlockPolicy(AllowAllPolicy):
            def before_execute(self, context):
                raise PolicyViolationError("blocked")

        h = DefaultAgentHarness(MockRuntime([ok_result()]), policy=BlockPolicy())
        with pytest.raises(PolicyViolationError):
            h.execute("x")


# ---------------------------------------------------------------- 14. Tool 注册/执行


class TestToolIntegration:
    def test_tool_via_harness_side_manager(self):
        tm = RegistryToolManager()
        tm.register("add", lambda a=0, b=0: a + b)
        assert tm.execute("add", a=2, b=3) == 5


# ---------------------------------------------------------------- 15. Context 更新


class TestContextUpdateThroughHarness:
    def test_context_carries_runtime_state(self):
        h = DefaultAgentHarness(simple_runtime())
        result = h.execute("同步状态")
        assert result.context.runtime_context["final_output"] == "同步状态"


# ---------------------------------------------------------------- 16. Memory 保存


class TestMemorySaveThroughHarness:
    def test_memory_saved_per_session(self):
        h = DefaultAgentHarness(MockRuntime([ok_result("o1"), ok_result("o2")]))
        h.execute(HarnessRequest(user_input="q1", session_id="s1"))
        h.execute(HarnessRequest(user_input="q2", session_id="s1"))
        records = h.memory_manager.get("s1")
        assert len(records) == 2
        assert records[0]["output"] == "o1"
        assert records[1]["user_input"] == "q2"

    def test_memory_saved_on_failure(self):
        h = DefaultAgentHarness(MockRuntime([fail_result()]))
        h.execute(HarnessRequest(user_input="q", session_id="s2"))
        records = h.memory_manager.get("s2")
        assert len(records) == 1
        assert records[0]["success"] is False


# ---------------------------------------------------------------- 最小 Demo（规格十九）


class TestMinimalDemo:
    def test_m06_demo_graph_through_harness(self):
        """DefaultAgentHarness -> LangGraphRuntime -> Input/RAG/LLM/Output。"""
        graph = build_rag_chat_graph(
            rag_service=MockRAGService(),
            llm=FakeLLM("基于知识的回答"),
            knowledge_base="kb1",
            prompt_template="资料:{context}\n问题:{user_input}",
        )
        h = DefaultAgentHarness(LangGraphRuntime(graph))
        result = h.execute(HarnessRequest(user_input="什么是RAG", session_id="demo"))
        assert result.success is True
        assert result.output == "基于知识的回答"
        assert len(result.context.retrieved_documents) == 1
        assert h.memory_manager.get("demo")[0]["success"] is True


# ---------------------------------------------------------------- 异常体系


class TestHarnessExceptions:
    def test_exceptions_inherit_chatchat_error(self):
        assert issubclass(HarnessError, ChatchatError)
        assert issubclass(PolicyViolationError, HarnessError)
