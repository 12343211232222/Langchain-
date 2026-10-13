"""审计修复项回归测试（2026-10-12）。

覆盖：
- P0-3 M03/M04 provider 工厂
- P0-1 M06 LangGraphRuntime 异步路径（arun / astream）
- P1-6 GraphBuilder 泛型化（state_schema）
- P1-4 GRAPH_ENGINE 灰度开关 + 冒烟
"""

import pytest

from chatchat.agent.engine import (
    get_graph_engine,
    graph_engine_smoke,
    is_langgraph_enabled,
)
from chatchat.agent.runtime import (
    GraphBuilder,
    InputNode,
    LangGraphRuntime,
    OutputNode,
    PassthroughNode,
)


class TestGraphBuilderGenericState:
    def test_default_state_schema_unchanged(self):
        graph = (
            GraphBuilder()
            .add_node("in", InputNode())
            .set_entry("in")
            .build()
        )
        result = graph.invoke({"user_input": "x"})
        assert result["user_input"] == "x"

    def test_custom_state_schema(self):
        from typing import TypedDict

        from chatchat.agent.runtime.state import AgentState

        class MyState(AgentState, total=False):
            custom_field: str

        def write_custom(state):
            return {"custom_field": "ok"}

        from chatchat.agent.runtime.node import FunctionNode

        graph = (
            GraphBuilder(state_schema=MyState)
            .add_node("in", InputNode())
            .add_node("work", FunctionNode(write_custom))
            .set_entry("in")
            .add_edge("in", "work")
            .build()
        )
        result = graph.invoke({"user_input": "x"})
        assert result["custom_field"] == "ok"


class TestAsyncRuntime:
    @pytest.fixture
    def runtime(self):
        graph = (
            GraphBuilder()
            .add_node("in", InputNode())
            .add_node("out", OutputNode())
            .set_entry("in")
            .add_edge("in", "out")
            .build()
        )
        return LangGraphRuntime(graph)

    async def test_arun_success(self, runtime):
        result = await runtime.arun("hello")
        assert result.success is True
        assert result.output == "hello"

    async def test_arun_failure(self):
        def boom(state):
            raise RuntimeError("boom")

        from chatchat.agent.runtime.node import FunctionNode

        graph = (
            GraphBuilder()
            .add_node("in", InputNode())
            .add_node("bad", FunctionNode(boom))
            .set_entry("in")
            .add_edge("in", "bad")
            .build()
        )
        result = await LangGraphRuntime(graph).arun("x")
        assert result.success is False
        assert "boom" in result.error

    async def test_astream_yields_events(self, runtime):
        events = [e async for e in runtime.astream("hi")]
        assert events  # 至少有节点快照
        assert any(isinstance(e, dict) for e in events)

    async def test_interface_default_arun_fallback(self):
        """AgentRuntimeInterface.arun 默认回退同步 run。"""
        from chatchat.agent.runtime import AgentRuntimeInterface, RuntimeResult

        class SyncOnly(AgentRuntimeInterface):
            def run(self, input_data):
                return RuntimeResult(success=True, output="sync")

        result = await SyncOnly().arun("x")
        assert result.success is True
        assert result.output == "sync"


class TestGraphEngineSwitch:
    def test_default_engine_is_legacy(self, monkeypatch):
        from chatchat.settings import Settings

        monkeypatch.setattr(Settings.basic_settings, "GRAPH_ENGINE", "legacy")
        assert get_graph_engine() == "legacy"
        assert is_langgraph_enabled() is False

    def test_langgraph_engine(self, monkeypatch):
        from chatchat.settings import Settings

        monkeypatch.setattr(Settings.basic_settings, "GRAPH_ENGINE", "langgraph")
        assert get_graph_engine() == "langgraph"
        assert is_langgraph_enabled() is True

    def test_unknown_engine_fallback(self, monkeypatch):
        from chatchat.settings import Settings

        monkeypatch.setattr(Settings.basic_settings, "GRAPH_ENGINE", "nonsense")
        assert get_graph_engine() == "legacy"

    def test_smoke(self):
        assert graph_engine_smoke() is True


class TestProviders:
    @pytest.fixture(autouse=True)
    def isolate_switch(self, monkeypatch):
        """隔离 yaml 中 RAGFLOW_PARSER_ENABLED 真实值对 provider 选择的影响。"""
        from chatchat.settings import Settings
        import chatchat.rag.parser.provider as provider_mod

        monkeypatch.setattr(Settings.basic_settings, "RAGFLOW_PARSER_ENABLED", False)
        monkeypatch.setattr(provider_mod, "_parser", None)
        yield
        monkeypatch.setattr(provider_mod, "_parser", None)

    def test_document_parser_provider(self):
        from chatchat.rag.parser import CurrentDocumentParser, get_document_parser

        parser = get_document_parser()
        assert isinstance(parser, CurrentDocumentParser)
        assert get_document_parser() is parser  # 单例

    def test_chunker_provider(self):
        from chatchat.rag.chunk import CurrentChunker, get_chunker

        chunker = get_chunker()
        assert isinstance(chunker, CurrentChunker)
        assert get_chunker() is chunker
