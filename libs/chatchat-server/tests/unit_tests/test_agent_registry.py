"""M08 Agent Registry 注册与发现层测试。

全部使用 MockAgent / MockFactory / MockHarness / MockRuntime，
不调用真实 LLM。
"""

import pytest

from chatchat.agent.harness import (
    AgentHarnessInterface,
    DefaultAgentHarness,
    HarnessRequest,
    HarnessResult,
)
from chatchat.agent.registry import (
    AgentAlreadyExistsError,
    AgentDefinition,
    AgentFactory,
    AgentInterface,
    AgentNotFoundError,
    AgentRegistryError,
    AgentStatus,
    CallableAgentFactory,
    HarnessAgent,
    InMemoryAgentRegistry,
    InvalidAgentDefinitionError,
    InvalidAgentFactoryError,
)
from chatchat.agent.runtime import AgentRuntimeInterface, RuntimeResult
from chatchat.core.exceptions import ChatchatError


# ---------------------------------------------------------------- mocks


class MockRuntime(AgentRuntimeInterface):
    def __init__(self, output="mock output"):
        self.output = output
        self.calls = 0

    def run(self, input_data):
        self.calls += 1
        return RuntimeResult(
            success=True,
            output=self.output,
            state={"user_input": "q", "final_output": self.output,
                   "execution_status": "success"},
        )


class MockHarness(AgentHarnessInterface):
    def __init__(self, output="harness output"):
        self.output = output
        self.requests = []

    def execute(self, request):
        self.requests.append(request)
        return HarnessResult(success=True, output=self.output)


class MockAgent(AgentInterface):
    def __init__(self, definition, **kwargs):
        self._definition = definition
        self.kwargs = kwargs

    @property
    def agent_id(self):
        return self._definition.agent_id

    def execute(self, request):
        return HarnessResult(success=True, output="mock agent output")


class MockFactory(AgentFactory):
    def __init__(self):
        self.created_with = []

    def create(self, definition, **kwargs):
        self.created_with.append((definition.agent_id, kwargs))
        return MockAgent(definition, **kwargs)


def make_definition(agent_id="mock_agent", **overrides):
    params = dict(
        agent_id=agent_id,
        name="Mock Agent",
        description="for tests",
        version="0.1.0",
        metadata={"domain": "test"},
        tags=["mock"],
    )
    params.update(overrides)
    return AgentDefinition(**params)


# ---------------------------------------------------------------- 1-2. AgentDefinition


class TestAgentDefinition:
    def test_create_with_defaults(self):
        d = AgentDefinition(agent_id="a1")
        assert d.agent_id == "a1"
        assert d.version == "0.1.0"
        assert d.enabled is True
        assert d.metadata == {}

    def test_definition_validation(self):
        with pytest.raises(InvalidAgentDefinitionError):
            AgentDefinition(agent_id="")
        with pytest.raises(InvalidAgentDefinitionError):
            AgentDefinition(agent_id="   ")
        with pytest.raises(InvalidAgentDefinitionError):
            AgentDefinition(agent_id=None)
        with pytest.raises(InvalidAgentDefinitionError):
            AgentDefinition(agent_id="x", metadata=None)


# ---------------------------------------------------------------- 3-9. Registry 基本操作


class TestRegistry:
    def test_register_and_get(self):
        r = InMemoryAgentRegistry()
        r.register(make_definition(), MockFactory())
        assert r.get("mock_agent").name == "Mock Agent"

    def test_duplicate_register_raises(self):
        r = InMemoryAgentRegistry()
        r.register(make_definition(), MockFactory())
        with pytest.raises(AgentAlreadyExistsError):
            r.register(make_definition(), MockFactory())

    def test_get_missing_raises_not_found(self):
        r = InMemoryAgentRegistry()
        with pytest.raises(AgentNotFoundError):
            r.get("nope")

    def test_exists(self):
        r = InMemoryAgentRegistry()
        assert r.exists("mock_agent") is False
        r.register(make_definition(), MockFactory())
        assert r.exists("mock_agent") is True

    def test_list(self):
        r = InMemoryAgentRegistry()
        r.register(make_definition("a1"), MockFactory())
        r.register(make_definition("a2"), MockFactory())
        ids = {d.agent_id for d in r.list()}
        assert ids == {"a1", "a2"}

    def test_unregister(self):
        r = InMemoryAgentRegistry()
        r.register(make_definition(), MockFactory())
        r.unregister("mock_agent")
        assert r.exists("mock_agent") is False
        with pytest.raises(AgentNotFoundError):
            r.unregister("mock_agent")

    def test_invalid_definition_and_factory(self):
        r = InMemoryAgentRegistry()
        with pytest.raises(InvalidAgentDefinitionError):
            r.register("not-a-definition", MockFactory())
        with pytest.raises(InvalidAgentFactoryError):
            r.register(make_definition(), object())


# ---------------------------------------------------------------- 10-11. Factory


class TestFactory:
    def test_factory_create(self):
        factory = MockFactory()
        agent = factory.create(make_definition())
        assert isinstance(agent, AgentInterface)
        assert agent.agent_id == "mock_agent"

    def test_factory_receives_kwargs(self):
        factory = MockFactory()
        agent = factory.create(make_definition(), model="m1", temperature=0.1)
        assert agent.kwargs == {"model": "m1", "temperature": 0.1}
        assert factory.created_with[0][1]["model"] == "m1"

    def test_callable_factory_wrapper(self):
        factory = CallableAgentFactory(lambda definition, **kw: MockAgent(definition))
        agent = factory.create(make_definition("wrapped"))
        assert agent.agent_id == "wrapped"


# ---------------------------------------------------------------- 12. 多 Agent 注册


class TestMultiAgent:
    def test_multiple_agents_independent(self):
        r = InMemoryAgentRegistry()
        r.register(make_definition("a1", name="A1"), MockFactory())
        r.register(make_definition("a2", name="A2"), MockFactory())
        assert r.get("a1").name == "A1"
        assert r.get("a2").name == "A2"
        r.unregister("a1")
        assert r.exists("a2") is True


# ---------------------------------------------------------------- 13-14. metadata / version


class TestMetadataAndVersion:
    def test_metadata_preserved(self):
        r = InMemoryAgentRegistry()
        r.register(make_definition(metadata={"domain": "tender", "owner": "x"}),
                   MockFactory())
        assert r.get("mock_agent").metadata["domain"] == "tender"

    def test_version_preserved(self):
        r = InMemoryAgentRegistry()
        r.register(make_definition(version="2.3.4"), MockFactory())
        assert r.get("mock_agent").version == "2.3.4"


# ---------------------------------------------------------------- 15. MockAgent 创建


class TestMockAgentCreation:
    def test_registry_create_returns_mock_agent(self):
        r = InMemoryAgentRegistry()
        r.register(make_definition(), MockFactory())
        agent = r.create("mock_agent", custom="v")
        assert isinstance(agent, MockAgent)
        assert agent.kwargs == {"custom": "v"}

    def test_create_missing_raises(self):
        r = InMemoryAgentRegistry()
        with pytest.raises(AgentNotFoundError):
            r.create("ghost")


# ---------------------------------------------------------------- 16. Registry + Factory 集成


class TestRegistryFactoryIntegration:
    def test_create_delegates_to_factory(self):
        factory = MockFactory()
        r = InMemoryAgentRegistry()
        r.register(make_definition(), factory)
        agent = r.create("mock_agent", k=1)
        assert factory.created_with == [("mock_agent", {"k": 1})]
        assert agent.agent_id == "mock_agent"


# ---------------------------------------------------------------- 17. Registry + M07 Harness 集成


class TestHarnessIntegration:
    def test_registry_to_harness_to_runtime_chain(self):
        """Registry → HarnessAgent → DefaultAgentHarness → MockRuntime 接线。"""
        runtime = MockRuntime(output="runtime ok")

        def build(definition, **kwargs):
            return HarnessAgent(definition, DefaultAgentHarness(runtime))

        r = InMemoryAgentRegistry()
        r.register(make_definition("rag_demo"), CallableAgentFactory(build))

        agent = r.create("rag_demo")
        assert isinstance(agent, HarnessAgent)

        result = agent.execute(HarnessRequest(user_input="问题", session_id="s1"))
        assert result.success is True
        assert result.output == "runtime ok"
        assert runtime.calls == 1

    def test_harness_agent_accepts_plain_string(self):
        harness = MockHarness()
        agent = HarnessAgent(make_definition("h1"), harness)
        result = agent.execute("你好")
        assert result.success is True
        assert harness.requests == ["你好"]


# ---------------------------------------------------------------- 生命周期与异常


class TestLifecycleAndExceptions:
    def test_status_transitions(self):
        r = InMemoryAgentRegistry()
        r.register(make_definition(), MockFactory())
        assert r._registrations["mock_agent"].status == AgentStatus.REGISTERED
        r.create("mock_agent")
        assert r._registrations["mock_agent"].status == AgentStatus.ENABLED

    def test_exceptions_inherit_chatchat_error(self):
        assert issubclass(AgentRegistryError, ChatchatError)
        assert issubclass(AgentAlreadyExistsError, AgentRegistryError)
        assert issubclass(AgentNotFoundError, AgentRegistryError)
        assert issubclass(InvalidAgentDefinitionError, AgentRegistryError)
        assert issubclass(InvalidAgentFactoryError, AgentRegistryError)
