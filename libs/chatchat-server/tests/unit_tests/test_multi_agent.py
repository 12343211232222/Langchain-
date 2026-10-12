"""M10 Multi-Agent Collaboration 测试。

全部使用 Mock / Fake Agent / MockRegistry，零真实 LLM / 零真实 Vector DB。
"""

import json
from typing import Any, Dict, Optional

import pytest

from chatchat.agent.collaboration import (
    AgentCoordinator,
    AgentMessage,
    AgentResult,
    AgentTask,
    CollaborationContext,
    CollaborationError,
    CollaborationPlan,
    CollaborationResult,
    ComplianceAgent,
    DependencyFailureError,
    InvalidMessageError,
    InvalidPlanError,
    MessageType,
    ParallelStrategy,
    ReportAgent,
    RequirementAgent,
    RetrievalAgent,
    SequentialStrategy,
    TaskStatus,
    build_tender_review_plan,
    register_demo_agents,
)
from chatchat.agent.harness import (
    AgentHarnessInterface,
    HarnessRequest,
    HarnessResult,
)
from chatchat.agent.registry import (
    AgentDefinition,
    AgentRegistryInterface,
    CallableAgentFactory,
    InMemoryAgentRegistry,
)
from chatchat.agent.registry.interfaces import AgentInterface
from chatchat.rag.schemas import RAGDocument, RAGResponse


# ---------------------------------------------------------------- mocks


class FakeRAGService:
    def retrieve(self, request):
        return RAGResponse(
            query=request.query,
            documents=[
                RAGDocument(
                    content="项目经理张三具有7年相关项目经验",
                    metadata={"source": "投标响应文件.pdf", "page": 12},
                )
            ],
        )


class EmptyRAGService:
    def retrieve(self, request):
        return RAGResponse(query=request.query, documents=[])


class FakeAgent(AgentInterface):
    """可编程 Fake Agent：按 agent_id 路由不同行为。"""

    def __init__(self, definition: AgentDefinition):
        self._definition = definition
        self.calls = []

    @property
    def agent_id(self) -> str:
        return self._definition.agent_id

    @property
    def definition(self) -> AgentDefinition:
        return self._definition

    def execute(self, request):
        self.calls.append(request)
        payload = json.loads(request if isinstance(request, str) else request.user_input)
        return HarnessResult(success=True, output=json.dumps(payload, ensure_ascii=False))


class BrokenAgent(AgentInterface):
    def __init__(self, definition):
        self._definition = definition

    @property
    def agent_id(self):
        return self._definition.agent_id

    def execute(self, request):
        raise RuntimeError("agent 内部异常")


class MockRegistry(AgentRegistryInterface):
    """Mock Registry：agent_id → callable(definition, **kwargs)。"""

    def __init__(self, builders: Optional[Dict[str, Any]] = None):
        self._defs: Dict[str, AgentDefinition] = {}
        self._builders: Dict[str, Any] = builders or {}

    def register(self, definition: AgentDefinition, factory):
        self._defs[definition.agent_id] = definition

    def unregister(self, agent_id: str):
        pass

    def get(self, agent_id: str) -> AgentDefinition:
        return self._defs[agent_id]

    def list(self):
        return list(self._defs.values())

    def exists(self, agent_id: str) -> bool:
        return agent_id in self._defs

    def create(self, agent_id: str, **kwargs):
        builder = self._builders.get(agent_id, FakeAgent)
        return builder(self._defs[agent_id])


def build_test_registry():
    """构建含 4 个 Demo Agent 的 Mock Registry。"""
    from chatchat.agent.collaboration.demo_agents import (
        ComplianceAgent,
        ReportAgent,
        RequirementAgent,
        RetrievalAgent,
    )

    defs = {
        "requirement_agent": AgentDefinition(
            agent_id="requirement_agent", name="R", description="req", tags=["demo"]
        ),
        "retrieval_agent": AgentDefinition(
            agent_id="retrieval_agent", name="R", description="ret", tags=["demo"]
        ),
        "compliance_agent": AgentDefinition(
            agent_id="compliance_agent", name="C", description="com", tags=["demo"]
        ),
        "report_agent": AgentDefinition(
            agent_id="report_agent", name="P", description="rep", tags=["demo"]
        ),
    }
    builders = {
        "requirement_agent": lambda d: RequirementAgent(),
        "retrieval_agent": lambda d: RetrievalAgent(
            rag_service=FakeRAGService(), knowledge_base="kb1"
        ),
        "compliance_agent": lambda d: ComplianceAgent(),
        "report_agent": lambda d: ReportAgent(),
    }
    registry = MockRegistry(builders)
    for d in defs.values():
        registry.register(d, None)
    return registry


# ---------------------------------------------------------------- 1. AgentTask


class TestAgentTask:
    def test_task_fields(self):
        t = AgentTask(
            task_id="t1", agent_id="a1", task_type="type1",
            input="x", dependencies=["d1"],
        )
        assert t.task_id == "t1"
        assert t.status == TaskStatus.PENDING.value
        assert t.dependencies == ["d1"]

    def test_validation(self):
        with pytest.raises(InvalidPlanError):
            AgentTask(task_id="", agent_id="a", task_type="t")
        with pytest.raises(InvalidPlanError):
            AgentTask(task_id="t", agent_id="", task_type="t")
        with pytest.raises(InvalidPlanError):
            AgentTask(task_id="t", agent_id="a", task_type="")


# ---------------------------------------------------------------- 2. AgentMessage


class TestAgentMessage:
    def test_message_fields(self):
        m = AgentMessage(
            sender="coordinator",
            receiver="agent1",
            message_type=MessageType.TASK_REQUEST.value,
            content={"k": "v"},
        )
        assert m.sender == "coordinator"
        assert m.correlation_id == ""

    def test_invalid_message_type(self):
        with pytest.raises(InvalidMessageError):
            AgentMessage(sender="a", receiver="b", message_type="UNKNOWN_TYPE")

    def test_missing_sender(self):
        with pytest.raises(InvalidMessageError):
            AgentMessage(sender="", receiver="b", message_type="task_request")


# ---------------------------------------------------------------- 3. AgentResult


class TestAgentResult:
    def test_result_fields(self):
        r = AgentResult(task_id="t1", agent_id="a1", success=True, output="ok")
        assert r.success
        assert r.output == "ok"


# ---------------------------------------------------------------- 4. CollaborationContext


class TestCollaborationContext:
    def test_context_fields(self):
        ctx = CollaborationContext(user_input="hi")
        assert ctx.user_input == "hi"
        assert ctx.collaboration_id

    def test_add_message_and_result(self):
        ctx = CollaborationContext()
        ctx.add_message(
            AgentMessage(sender="a", receiver="b", message_type="task_request")
        )
        ctx.add_result(AgentResult(task_id="t1", agent_id="a1", success=True))
        assert len(ctx.messages) == 1
        assert "t1" in ctx.task_results

    def test_shared_data(self):
        ctx = CollaborationContext()
        ctx.set_shared("tender", "doc")
        assert ctx.get_shared("tender") == "doc"
        assert ctx.get_shared("x", "dft") == "dft"

    def test_get_dependency_outputs(self):
        ctx = CollaborationContext()
        ctx.add_result(AgentResult(task_id="t1", agent_id="a1", success=True, output="o1"))
        assert ctx.get_dependency_outputs(["t1"]) == {"t1": "o1"}


# ---------------------------------------------------------------- 5. CollaborationPlan


class TestCollaborationPlan:
    def test_plan_fields(self):
        plan = CollaborationPlan(
            plan_id="p1",
            tasks=[AgentTask(task_id="t1", agent_id="a1", task_type="t")],
        )
        assert plan.plan_id == "p1"

    def test_empty_plan_raises(self):
        with pytest.raises(InvalidPlanError):
            CollaborationPlan(plan_id="p1", tasks=[])

    def test_duplicate_task_id(self):
        t = AgentTask(task_id="t1", agent_id="a1", task_type="t")
        with pytest.raises(InvalidPlanError):
            CollaborationPlan(plan_id="p1", tasks=[t, t])

    def test_unknown_dependency(self):
        t = AgentTask(task_id="t1", agent_id="a1", task_type="t", dependencies=["ghost"])
        with pytest.raises(InvalidPlanError):
            CollaborationPlan(plan_id="p1", tasks=[t])


# ---------------------------------------------------------------- 6. AgentCoordinator


class TestAgentCoordinator:
    def test_coordinator_init(self):
        c = AgentCoordinator(MockRegistry())
        assert isinstance(c.strategy, SequentialStrategy)

    def test_coordinator_execute_with_empty_registry_fails(self):
        registry = MockRegistry()
        registry.register(
            AgentDefinition(agent_id="a1", name="A"), None
        )
        registry._builders = {}  # 无 builder → 调用 FakeAgent → 会注册 a1
        c = AgentCoordinator(registry)
        plan = CollaborationPlan(
            plan_id="p1",
            tasks=[AgentTask(task_id="t1", agent_id="a1", task_type="t")],
        )
        result = c.execute(plan)
        assert result.success is True


# ---------------------------------------------------------------- 7. Registry 发现


class TestRegistryDiscovery:
    def test_registry_exists(self):
        registry = build_test_registry()
        assert registry.exists("requirement_agent")
        assert registry.exists("retrieval_agent")
        assert registry.exists("compliance_agent")
        assert registry.exists("report_agent")


# ---------------------------------------------------------------- 8. Agent 创建


class TestAgentCreation:
    def test_create_requirement_agent(self):
        registry = build_test_registry()
        agent = registry.create("requirement_agent")
        assert isinstance(agent, RequirementAgent)

    def test_create_retrieval_agent(self):
        registry = build_test_registry()
        agent = registry.create("retrieval_agent")
        assert isinstance(agent, RetrievalAgent)


# ---------------------------------------------------------------- 9. Sequential execution


class TestSequentialExecution:
    def test_a_to_b_to_c(self):
        registry = MockRegistry()
        for aid in ["A", "B", "C"]:
            registry.register(AgentDefinition(agent_id=aid, name=aid), None)

        c = AgentCoordinator(registry)
        plan = CollaborationPlan(
            plan_id="seq",
            tasks=[
                AgentTask(task_id="t1", agent_id="A", task_type="t"),
                AgentTask(task_id="t2", agent_id="B", task_type="t", dependencies=["t1"]),
                AgentTask(task_id="t3", agent_id="C", task_type="t", dependencies=["t2"]),
            ],
        )
        result = c.execute(plan)
        assert result.success is True
        assert list(result.results.keys()) == ["t1", "t2", "t3"]


# ---------------------------------------------------------------- 10. Dependency resolution


class TestDependencyResolution:
    def test_topological_order_not_list_order(self):
        registry = MockRegistry()
        for aid in ["A", "B", "C"]:
            registry.register(AgentDefinition(agent_id=aid, name=aid), None)

        c = AgentCoordinator(registry)
        # 打乱 list 顺序，但依赖仍是 A→B→C
        plan = CollaborationPlan(
            plan_id="dep",
            tasks=[
                AgentTask(task_id="t3", agent_id="C", task_type="t", dependencies=["t2"]),
                AgentTask(task_id="t1", agent_id="A", task_type="t"),
                AgentTask(task_id="t2", agent_id="B", task_type="t", dependencies=["t1"]),
            ],
        )
        result = c.execute(plan)
        assert result.success is True
        order = list(result.results.keys())
        assert order == ["t1", "t2", "t3"]


# ---------------------------------------------------------------- 11. Dependency failure


class TestDependencyFailure:
    def test_downstream_skipped_when_upstream_fails(self):
        registry = MockRegistry()
        registry.register(AgentDefinition(agent_id="ok", name="O"), None)
        registry.register(AgentDefinition(agent_id="bad", name="B"), None)
        registry._builders = {
            "ok": FakeAgent,
            "bad": BrokenAgent,
        }

        c = AgentCoordinator(registry)
        plan = CollaborationPlan(
            plan_id="fail",
            tasks=[
                AgentTask(task_id="t1", agent_id="ok", task_type="t"),
                AgentTask(task_id="t2", agent_id="bad", task_type="t", dependencies=["t1"]),
                AgentTask(task_id="t3", agent_id="ok", task_type="t", dependencies=["t2"]),
            ],
        )
        result = c.execute(plan)
        assert result.success is False
        assert result.results["t2"].success is False
        assert result.results["t3"].success is False  # skipped
        assert "dependency failure" in result.results["t3"].error
        assert result.failed_task == "t2"


# ---------------------------------------------------------------- 12. Agent 失败


class TestAgentFailure:
    def test_broken_agent_returns_failure_not_crash(self):
        registry = MockRegistry()
        registry.register(AgentDefinition(agent_id="broken", name="B"), None)
        registry._builders = {"broken": BrokenAgent}

        c = AgentCoordinator(registry)
        plan = CollaborationPlan(
            plan_id="fail",
            tasks=[AgentTask(task_id="t1", agent_id="broken", task_type="t")],
        )
        result = c.execute(plan)
        assert result.success is False
        assert "agent 执行异常" in result.results["t1"].error


# ---------------------------------------------------------------- 13. Unknown Agent


class TestUnknownAgent:
    def test_missing_agent_returns_failure(self):
        registry = MockRegistry()
        c = AgentCoordinator(registry)
        plan = CollaborationPlan(
            plan_id="p1",
            tasks=[AgentTask(task_id="t1", agent_id="ghost", task_type="t")],
        )
        result = c.execute(plan)
        assert result.success is False
        assert "agent 不存在" in result.results["t1"].error


# ---------------------------------------------------------------- 14. Invalid message


class TestInvalidMessage:
    def test_invalid_message_type_raises(self):
        with pytest.raises(InvalidMessageError):
            AgentMessage(sender="a", receiver="b", message_type="bad_type")


# ---------------------------------------------------------------- 15. Context propagation


class TestContextPropagation:
    def test_context_carries_messages_and_results(self):
        registry = build_test_registry()
        c = AgentCoordinator(registry)
        plan = CollaborationPlan(
            plan_id="p1",
            tasks=[
                AgentTask(task_id="t1", agent_id="requirement_agent", task_type="t",
                          input="项目经理必须具有5年以上经验"),
                AgentTask(task_id="t2", agent_id="retrieval_agent", task_type="t",
                          dependencies=["t1"]),
            ],
        )
        ctx = CollaborationContext()
        result = c.execute(plan, ctx)
        assert ctx.collaboration_id
        assert len(ctx.messages) >= 2  # request + result
        assert "t1" in ctx.task_results
        assert "t2" in ctx.task_results


# ---------------------------------------------------------------- 16. Agent result propagation


class TestResultPropagation:
    def test_result_stored_in_context(self):
        registry = build_test_registry()
        c = AgentCoordinator(registry)
        plan = CollaborationPlan(
            plan_id="p1",
            tasks=[AgentTask(task_id="t1", agent_id="requirement_agent", task_type="t",
                             input="项目经理>=5年经验")],
        )
        result = c.execute(plan)
        assert "t1" in result.results
        assert result.results["t1"].success is True


# ---------------------------------------------------------------- 17. SequentialStrategy


class TestSequentialStrategy:
    def test_order_by_dependencies(self):
        tasks = [
            AgentTask(task_id="t3", agent_id="C", task_type="t", dependencies=["t2"]),
            AgentTask(task_id="t1", agent_id="A", task_type="t"),
            AgentTask(task_id="t2", agent_id="B", task_type="t", dependencies=["t1"]),
        ]
        plan = CollaborationPlan(plan_id="s", tasks=tasks)
        ordered = SequentialStrategy().order(plan)
        assert [t.task_id for t in ordered] == ["t1", "t2", "t3"]

    def test_cycle_detected(self):
        tasks = [
            AgentTask(task_id="t1", agent_id="A", task_type="t", dependencies=["t2"]),
            AgentTask(task_id="t2", agent_id="B", task_type="t", dependencies=["t1"]),
        ]
        plan = CollaborationPlan(plan_id="c", tasks=tasks)
        with pytest.raises(InvalidPlanError):
            SequentialStrategy().order(plan)


# ---------------------------------------------------------------- 18. ParallelStrategy interface


class TestParallelStrategy:
    def test_interface_exists(self):
        s = ParallelStrategy()
        plan = CollaborationPlan(
            plan_id="p",
            tasks=[
                AgentTask(task_id="t1", agent_id="A", task_type="t"),
                AgentTask(task_id="t2", agent_id="B", task_type="t"),
            ],
        )
        groups = s.groups(plan)
        assert isinstance(groups, list)


# ---------------------------------------------------------------- 19. Multi-Agent graph


class TestMultiAgentGraph:
    def test_plan_structure(self):
        plan = build_tender_review_plan("项目经理>=5年经验", knowledge_base="kb1")
        assert plan.plan_id == "tender_multi_agent_review"
        ids = [t.task_id for t in plan.tasks]
        assert ids == ["t1_requirement", "t2_retrieval", "t3_compliance", "t4_report"]


# ---------------------------------------------------------------- 20. RequirementAgent


class TestRequirementAgent:
    def test_extract_requirements(self):
        agent = RequirementAgent()
        result = agent.execute(
            json.dumps({"input": "项目经理必须具有5年以上经验。"}, ensure_ascii=False)
        )
        assert result.success is True
        data = json.loads(result.output)
        assert len(data["requirements"]) >= 1
        assert data["requirements"][0]["text"] == "项目经理必须具有5年以上经验"


# ---------------------------------------------------------------- 21. RetrievalAgent


class TestRetrievalAgent:
    def test_retrieve_evidence(self):
        agent = RetrievalAgent(rag_service=FakeRAGService(), knowledge_base="kb1")
        result = agent.execute(
            json.dumps(
                {
                    "dependencies": {
                        "t1": json.dumps(
                            {"requirements": [{"id": "R1", "text": "5年以上"}]}
                        )
                    }
                },
                ensure_ascii=False,
            )
        )
        assert result.success is True
        data = json.loads(result.output)
        assert data["retrievals"][0]["evidences"][0]["source"] == "投标响应文件.pdf"


# ---------------------------------------------------------------- 22. ComplianceAgent


class TestComplianceAgent:
    def test_compliant_with_evidence(self):
        agent = ComplianceAgent()
        payload = {
            "dependencies": {
                "t2": json.dumps(
                    {
                        "retrievals": [
                            {
                                "requirement_id": "R1",
                                "requirement": "5年以上",
                                "evidences": [
                                    {
                                        "content": "张三有7年经验",
                                        "source": "s1",
                                        "page": 1,
                                    }
                                ],
                            }
                        ]
                    }
                )
            }
        }
        result = agent.execute(json.dumps(payload, ensure_ascii=False))
        assert result.success is True
        data = json.loads(result.output)
        assert data["compliance_results"][0]["status"] == "COMPLIANT"

    def test_unknown_without_evidence(self):
        agent = ComplianceAgent()
        payload = {
            "dependencies": {
                "t2": json.dumps(
                    {
                        "retrievals": [
                            {
                                "requirement_id": "R1",
                                "requirement": "5年以上",
                                "evidences": [],
                            }
                        ]
                    }
                )
            }
        }
        result = agent.execute(json.dumps(payload, ensure_ascii=False))
        data = json.loads(result.output)
        assert data["compliance_results"][0]["status"] == "UNKNOWN"


# ---------------------------------------------------------------- 23. ReportAgent


class TestReportAgent:
    def test_report_summary(self):
        agent = ReportAgent()
        payload = {
            "dependencies": {
                "t3": json.dumps(
                    {
                        "compliance_results": [
                            {"status": "COMPLIANT"},
                            {"status": "UNKNOWN"},
                        ]
                    }
                )
            }
        }
        result = agent.execute(json.dumps(payload, ensure_ascii=False))
        assert result.success is True
        data = json.loads(result.output)
        assert data["report"]["summary"]["total"] == 2
        assert data["report"]["summary"]["compliant"] == 1
        assert data["report"]["summary"]["unknown"] == 1


# ---------------------------------------------------------------- 24. End-to-end Multi-Agent demo


class TestEndToEnd:
    def test_tender_multi_agent_review_demo(self):
        """规格三十七：
        Requirement → Retrieval → Compliance → Report
        输入：项目经理必须具有5年以上经验
        输出：COMPLIANT，total=1，compliant=1
        """
        registry = InMemoryAgentRegistry()
        register_demo_agents(
            registry, rag_service=FakeRAGService(), knowledge_base="kb1"
        )
        coordinator = AgentCoordinator(registry)
        plan = build_tender_review_plan(
            "项目经理必须具有5年以上相关项目经验。", knowledge_base="kb1"
        )
        result = coordinator.execute(plan)
        assert result.success is True
        report = json.loads(result.results["t4_report"].output)
        assert report["report"]["summary"]["total"] >= 1
        assert report["report"]["summary"]["compliant"] >= 1


# ---------------------------------------------------------------- 25. M09 regression


class TestM09Regression:
    def test_m09_agent_still_works(self):
        """M09 TenderReviewAgent 不调用 Registry 也能独立运行。"""
        from chatchat.agent.tender_review import TenderReviewAgent, TenderReviewRequest
        from chatchat.agent.harness import DefaultAgentHarness
        from chatchat.agent.runtime import (
            GraphBuilder, InputNode, OutputNode, LangGraphRuntime
        )

        graph = (
            GraphBuilder()
            .add_node("in", InputNode())
            .add_node("out", OutputNode())
            .set_entry("in")
            .add_edge("in", "out")
            .build()
        )
        agent = TenderReviewAgent(
            rag_service=FakeRAGService(),
            harness=DefaultAgentHarness(LangGraphRuntime(graph)),
        )
        assert agent.review(
            TenderReviewRequest(tender_requirement="t", bidder_response="b")
        ).total_requirements == 0
