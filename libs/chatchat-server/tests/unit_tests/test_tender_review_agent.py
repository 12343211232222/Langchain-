"""M09 Tender Review Agent 测试。

全部使用 Mock RAG / Mock LLM / Mock Harness，不调用真实 LLM / 不依赖真实 PDF。
"""

import json

import pytest

from chatchat.agent.harness import (
    AgentHarnessInterface,
    DefaultAgentHarness,
    HarnessRequest,
    HarnessResult,
)
from chatchat.agent.registry import (
    AgentDefinition,
    InMemoryAgentRegistry,
    CallableAgentFactory,
)
from chatchat.agent.runtime import (
    AgentRuntimeInterface,
    GraphBuilder,
    InputNode,
    LangGraphRuntime,
    OutputNode,
    RuntimeResult,
)
from chatchat.agent.tender_review import (
    ComplianceStatus,
    Evidence,
    RequirementCategory,
    RequirementReviewResult,
    TenderReviewAgent,
    TenderReviewError,
    TenderReviewRequest,
    TenderReviewResult,
    TenderReviewState,
    build_tender_review_graph,
    register_tender_review_agent,
    TENDER_REVIEW_AGENT_ID,
)
from chatchat.agent.tender_review.graph import build_tender_review_graph
from chatchat.agent.tender_review.nodes import (
    ComplianceAnalysisNode,
    RequirementClassificationNode,
    RequirementExtractionNode,
    ResponseMatchingNode,
    ResultAggregationNode,
    TenderInputNode,
    TenderRetrievalNode,
)
from chatchat.agent.tender_review.schemas import Requirement, RequirementCategory
from chatchat.agent.tender_review.state import TenderReviewState
from chatchat.rag.schemas import RAGDocument, RAGResponse


# ---------------------------------------------------------------- mocks


class FakeLLM:
    """可编程的 Fake LLM：invoke(messages) -> message。"""

    def __init__(self, reply="{}"):
        self.reply = reply
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[-1].content)
        return type("FakeMsg", (), {"content": self.reply})()


class FakeMessage:
    def __init__(self, content):
        self.content = content


class MockRAGService:
    """复用 M02 RAGServiceInterface 的 mock。"""

    def __init__(self, docs=None):
        self.docs = docs if docs is not None else [
            RAGDocument(
                content="项目经理张三具有7年相关项目经验",
                metadata={"source": "投标响应文件.pdf", "page": 12},
            )
        ]
        self.queries = []

    def retrieve(self, request):
        self.queries.append(request.query)
        return RAGResponse(query=request.query, documents=list(self.docs))


class MockHarness(AgentHarnessInterface):
    def __init__(self, output='{"results":[],"total_requirements":0}'):
        self.output = output
        self.requests = []

    def execute(self, request):
        self.requests.append(request)
        return HarnessResult(success=True, output=self.output)


def make_request(
    tender="项目经理必须具有 5 年以上相关项目经验。",
    response="项目经理张三具有 7 年相关项目经验。",
):
    return TenderReviewRequest(
        tender_requirement=tender,
        bidder_response=response,
        session_id="s1",
        metadata={"project": "p1"},
    )


# ---------------------------------------------------------------- 1. Request Schema


class TestRequestSchema:
    def test_request_fields(self):
        r = make_request()
        assert "5 年以上" in r.tender_requirement
        assert "张三" in r.bidder_response
        assert r.session_id == "s1"

    def test_request_defaults(self):
        r = TenderReviewRequest(tender_requirement="a")
        assert r.bidder_response == ""
        assert r.metadata == {}


# ---------------------------------------------------------------- 2. Result Schema


class TestResultSchema:
    def test_result_counts(self):
        result = TenderReviewResult(
            total_requirements=3,
            compliant_count=2,
            partial_count=1,
        )
        assert result.compliant_count == 2
        assert result.to_dict()["total_requirements"] == 3

    def test_requirement_review_result_fields(self):
        rr = RequirementReviewResult(
            requirement="r1",
            status=ComplianceStatus.COMPLIANT.value,
            evidence=[Evidence(content="e1", source="s1", page=10, score=0.9)],
            explanation="符合",
            risk="无",
        )
        assert rr.evidence[0].page == 10
        assert rr.status == "COMPLIANT"

    def test_result_from_dict(self):
        d = {
            "results": [
                {
                    "requirement": "r1",
                    "status": "COMPLIANT",
                    "matched_response": "resp",
                    "evidence": [{"content": "c1", "source": "s1", "page": None}],
                    "explanation": "ok",
                    "risk": "low",
                }
            ],
            "total_requirements": 1,
            "compliant_count": 1,
        }
        obj = TenderReviewResult.from_dict(d)
        assert obj.results[0].evidence[0].source == "s1"


# ---------------------------------------------------------------- 3. Requirement Extraction


class TestRequirementExtraction:
    def test_extraction_parses_json(self):
        fake = FakeLLM(
            '{"requirements": [{"id": "req_1", "text": "必须具有 ISO9001"}]}'
        )
        node = RequirementExtractionNode(llm=fake)
        update = node({"request": {"tender_requirement": "..."}})
        assert len(update["requirements"]) == 1
        assert update["requirements"][0]["text"] == "必须具有 ISO9001"

    def test_extraction_empty_requirement(self):
        fake = FakeLLM('{"requirements": []}')
        node = RequirementExtractionNode(llm=fake)
        update = node({"request": {"tender_requirement": ""}})
        assert update["requirements"] == []

    def test_extraction_llm_failure_graceful(self):
        fake = FakeLLM('not-json')
        node = RequirementExtractionNode(llm=fake)
        update = node({"request": {"tender_requirement": "x"}})
        assert "error" in update


# ---------------------------------------------------------------- 4. Requirement Classification


class TestRequirementClassification:
    def test_classify_parses_json(self):
        fake = FakeLLM(
            '{"classifications": [{"id": "req_1", "category": "qualification"}]}'
        )
        node = RequirementClassificationNode(llm=fake)
        update = node(
            {"requirements": [{"id": "req_1", "text": "t"}]}
        )
        assert update["requirements"][0]["category"] == "qualification"

    def test_classify_fallback_other(self):
        fake = FakeLLM('{"classifications": [{"id": "req_1", "category": "nonsense"}]}')
        node = RequirementClassificationNode(llm=fake)
        update = node({"requirements": [{"id": "req_1", "text": "t"}]})
        assert update["requirements"][0]["category"] == RequirementCategory.OTHER.value


# ---------------------------------------------------------------- 5. Retrieval Node


class TestRetrievalNode:
    def test_retrieval_calls_rag_service(self):
        rag = MockRAGService()
        node = TenderRetrievalNode(rag_service=rag, knowledge_base="kb1", top_k=3)
        update = node(
            {
                "request": {"knowledge_base": "kb1"},
                "requirements": [{"id": "req_1", "text": "项目经理经验"}],
            }
        )
        assert rag.queries == ["项目经理经验"]
        docs = update["retrieved_documents"][0]["documents"]
        assert docs[0]["source"] == "投标响应文件.pdf"
        assert docs[0]["page"] == 12

    def test_retrieval_empty_requirements(self):
        rag = MockRAGService()
        node = TenderRetrievalNode(rag_service=rag)
        update = node({"request": {}, "requirements": []})
        assert update["retrieved_documents"] == []

    def test_retrieval_rag_exception_graceful(self):
        class BadRAG:
            def retrieve(self, request):
                raise RuntimeError("boom")

        node = TenderRetrievalNode(rag_service=BadRAG(), knowledge_base="kb1")
        update = node(
            {
                "request": {},
                "requirements": [{"id": "req_1", "text": "x"}],
            }
        )
        assert update["retrieved_documents"][0]["documents"] == []


# ---------------------------------------------------------------- 6. Empty Retrieval


class TestEmptyRetrieval:
    def test_empty_documents_produces_empty_evidence(self):
        rag = MockRAGService(docs=[])
        node = TenderRetrievalNode(rag_service=rag, knowledge_base="kb1")
        update = node(
            {
                "request": {},
                "requirements": [{"id": "req_1", "text": "x"}],
            }
        )
        assert update["retrieved_documents"][0]["documents"] == []


# ---------------------------------------------------------------- 7. Response Matching


class TestResponseMatching:
    def test_match_finds_text(self):
        fake = FakeLLM("NONE")
        node = ResponseMatchingNode(llm=fake)
        update = node(
            {
                "request": {"bidder_response": "项目经理张三具有7年经验"},
                "requirements": [{"id": "req_1", "text": "项目经理张三具有7年经验"}],
            }
        )
        assert update["matched_response"][0]["text"] == "项目经理张三具有7年经验"

    def test_match_not_found(self):
        fake = FakeLLM("NONE")
        node = ResponseMatchingNode(llm=fake)
        update = node(
            {
                "request": {"bidder_response": "无关内容"},
                "requirements": [{"id": "req_1", "text": "ISO9001"}],
            }
        )
        assert update["matched_response"][0]["text"] is None


# ---------------------------------------------------------------- 8. Compliance Analysis


class TestComplianceAnalysis:
    def test_analyze_compliant(self):
        fake = FakeLLM(
            '{"status": "COMPLIANT", "explanation": "满足要求", "risk": ""}'
        )
        node = ComplianceAnalysisNode(llm=fake)
        update = node(
            {
                "request": {},
                "requirements": [{"id": "req_1", "text": "经验>=5年"}],
                "retrieved_documents": [
                    {
                        "requirement_id": "req_1",
                        "documents": [
                            {"content": "7年经验", "source": "s1", "page": 12}
                        ],
                    }
                ],
                "matched_response": [{"requirement_id": "req_1", "text": "7年经验"}],
            }
        )
        r = update["review_results"][0]
        assert r["status"] == ComplianceStatus.COMPLIANT.value
        assert r["evidence"][0]["source"] == "s1"
        assert r["evidence"][0]["page"] == 12

    def test_analyze_unknown_when_no_evidence(self):
        fake = FakeLLM(
            '{"status": "COMPLIANT", "explanation": "ok", "risk": ""}'
        )
        node = ComplianceAnalysisNode(llm=fake)
        update = node(
            {
                "request": {},
                "requirements": [{"id": "req_1", "text": "x"}],
                "retrieved_documents": [
                    {"requirement_id": "req_1", "documents": []}
                ],
                "matched_response": [{"requirement_id": "req_1", "text": None}],
            }
        )
        assert update["review_results"][0]["status"] == ComplianceStatus.UNKNOWN.value


# ---------------------------------------------------------------- 9. UNKNOWN 情况


class TestUnknownCompliance:
    def test_forced_unknown_bypasses_llm(self):
        fake = FakeLLM('{"status": "COMPLIANT"}')
        node = ComplianceAnalysisNode(llm=fake)
        update = node(
            {
                "request": {},
                "requirements": [{"id": "req_1", "text": "x"}],
                "retrieved_documents": [
                    {"requirement_id": "req_1", "documents": []}
                ],
                "matched_response": [{"requirement_id": "req_1", "text": None}],
            }
        )
        assert update["review_results"][0]["status"] == ComplianceStatus.UNKNOWN.value
        assert "证据不足" in update["review_results"][0]["explanation"]


# ---------------------------------------------------------------- 10. Result Aggregation


class TestResultAggregation:
    def test_counts_are_correct(self):
        node = ResultAggregationNode()
        update = node(
            {
                "review_results": [
                    {"status": "COMPLIANT"},
                    {"status": "NON_COMPLIANT"},
                    {"status": "UNKNOWN"},
                ]
            }
        )
        assert update["final_output"]
        assert update["execution_status"] == "success"
        data = json.loads(update["final_output"])
        assert data["compliant_count"] == 1
        assert data["non_compliant_count"] == 1
        assert data["unknown_count"] == 1


# ---------------------------------------------------------------- 11. Graph 构建


class TestGraphBuild:
    def test_graph_compiles(self):
        graph = build_tender_review_graph(rag_service=MockRAGService())
        assert graph is not None

    def test_graph_execution_end_to_end_mock(self):
        """全链路 mock：注入 FakeLLM 使各节点产生结构化输出。"""
        rag = MockRAGService()
        fake_llm = FakeLLM(
            '{"requirements": [{"id": "req_1", "text": "经验>=5年"}],'
            ' "classifications": [{"id": "req_1", "category": "experience"}],'
            ' "status": "COMPLIANT", "explanation": "满足", "risk": ""}'
        )
        graph = build_tender_review_graph(rag_service=rag, llm=fake_llm, knowledge_base="kb1")
        request = json.dumps(
            {
                "tender_requirement": "项目经理必须具有5年以上经验",
                "bidder_response": "项目经理张三具有7年经验",
                "knowledge_base": "kb1",
                "session_id": "s1",
            },
            ensure_ascii=False,
        )
        final = graph.invoke({"user_input": request})
        assert final["execution_status"] == "success"
        assert final.get("final_output")


# ---------------------------------------------------------------- 12. Graph 执行（更细）


class TestGraphExecution:
    def test_tender_input_node(self):
        node = TenderInputNode()
        update = node(
            {"user_input": '{"tender_requirement": "x", "bidder_response": "y"}'}
        )
        assert update["request"]["tender_requirement"] == "x"

    def test_tender_input_fallback_plain_text(self):
        node = TenderInputNode()
        update = node({"user_input": "纯文本输入"})
        assert update["request"]["tender_requirement"] == "纯文本输入"


# ---------------------------------------------------------------- 13. Agent.execute


class TestAgentExecute:
    def test_execute_with_mock_harness(self):
        expected = json.dumps(
            {"results": [], "total_requirements": 0, "compliant_count": 0}
        )
        agent = TenderReviewAgent(
            rag_service=MockRAGService(),
            harness=MockHarness(output=expected),
        )
        result = agent.execute(make_request())
        assert result.success is True
        assert result.output == expected

    def test_review_returns_structured_result(self):
        result_dict = {
            "results": [
                {
                    "requirement": "r1",
                    "status": "COMPLIANT",
                    "matched_response": "resp",
                    "evidence": [{"content": "c", "source": "s", "page": 1}],
                    "explanation": "ok",
                    "risk": "",
                }
            ],
            "total_requirements": 1,
            "compliant_count": 1,
        }
        agent = TenderReviewAgent(
            rag_service=MockRAGService(),
            harness=MockHarness(output=json.dumps(result_dict, ensure_ascii=False)),
        )
        review = agent.review(make_request())
        assert isinstance(review, TenderReviewResult)
        assert review.results[0].status == ComplianceStatus.COMPLIANT.value
        assert review.results[0].evidence[0].source == "s"

    def test_rag_service_injection_required(self):
        with pytest.raises(TenderReviewError):
            TenderReviewAgent(rag_service=None)


# ---------------------------------------------------------------- 14. Registry 注册


class TestRegistryRegistration:
    def test_register_and_create(self):
        registry = InMemoryAgentRegistry()
        register_tender_review_agent(registry, rag_service=MockRAGService())
        agent = registry.create("tender_review")
        assert agent.agent_id == "tender_review"

    def test_registry_definition_fields(self):
        registry = InMemoryAgentRegistry()
        register_tender_review_agent(registry, rag_service=MockRAGService())
        d = registry.get("tender_review")
        assert d.tags == ["tender", "review", "compliance"]
        assert d.metadata["domain"] == "tender"


# ---------------------------------------------------------------- 15. Registry 创建 Agent


class TestRegistryCreateAgent:
    def test_registry_kwargs_passed(self):
        registry = InMemoryAgentRegistry()
        register_tender_review_agent(
            registry, rag_service=MockRAGService(), knowledge_base="kb1"
        )
        agent = registry.create("tender_review", knowledge_base="kb2")
        # Agent 初始化不接受透传 kwargs 修改已有参数；工厂把 kwargs 传给 TenderReviewAgent
        # TenderReviewAgent.__init__ 接受 knowledge_base 透传
        assert isinstance(agent, TenderReviewAgent)


# ---------------------------------------------------------------- 16. Harness 集成


class TestHarnessIntegration:
    def test_agent_to_harness_chain(self):
        """TenderReviewAgent -> DefaultAgentHarness -> LangGraphRuntime。"""
        rag = MockRAGService()
        fake_llm = FakeLLM(
            '{"requirements": [{"id": "req_1", "text": "经验>=5年"}],'
            ' "classifications": [{"id": "req_1", "category": "experience"}],'
            ' "status": "COMPLIANT", "explanation": "满足", "risk": ""}'
        )
        agent = TenderReviewAgent(
            rag_service=rag,
            llm=fake_llm,
            knowledge_base="kb1",
        )
        result = agent.execute(make_request())
        assert result.success is True
        assert result.output
        assert isinstance(result, HarnessResult)


# ---------------------------------------------------------------- 17. RAGService Mock


class TestRAGServiceMock:
    def test_mock_rag_retrieve(self):
        rag = MockRAGService()
        from chatchat.rag.schemas import RAGQuery

        resp = rag.retrieve(RAGQuery(query="x", knowledge_base="kb1"))
        assert resp.documents[0].content == "项目经理张三具有7年相关项目经验"


# ---------------------------------------------------------------- 18. LLM Mock


class TestLLMMock:
    def test_fake_llm_invoked(self):
        fake = FakeLLM('{"a": 1}')
        assert fake.invoke([type("Msg", (), {"content": "hi"})()]).content == '{"a": 1}'


# ---------------------------------------------------------------- 19. 错误处理


class TestErrorHandling:
    def test_agent_review_failure(self):
        agent = TenderReviewAgent(
            rag_service=MockRAGService(),
            harness=MockHarness(output=None),
        )
        with pytest.raises(TenderReviewError):
            agent.review(make_request())

    def test_review_success_on_valid_output(self):
        d = {
            "results": [],
            "total_requirements": 0,
            "compliant_count": 0,
            "partial_count": 0,
            "non_compliant_count": 0,
            "unknown_count": 0,
        }
        agent = TenderReviewAgent(
            rag_service=MockRAGService(),
            harness=MockHarness(output=json.dumps(d)),
        )
        assert agent.review(make_request()).total_requirements == 0


# ---------------------------------------------------------------- 20. 最小端到端 Demo


class TestMinimalDemo:
    def test_minimal_end_to_end(self):
        """规格二十七：
        招标要求 + 投标响应 + mock RAG -> COMPLIANT。
        """
        rag = MockRAGService()
        fake_llm = FakeLLM(
            '{"requirements": [{"id": "req_1", "text": "项目经理>=5年经验"}],'
            ' "classifications": [{"id": "req_1", "category": "experience"}],'
            ' "status": "COMPLIANT", "explanation": "张三7年经验满足要求", "risk": ""}'
        )
        agent = TenderReviewAgent(
            rag_service=rag,
            llm=fake_llm,
            knowledge_base="kb1",
        )
        review = agent.review(make_request())
        assert review.total_requirements >= 1
        result = review.results[0]
        assert result.status == ComplianceStatus.COMPLIANT.value
        assert result.evidence[0].source == "投标响应文件.pdf"
        assert result.evidence[0].page == 12


# ---------------------------------------------------------------- 21. TenderReviewState


class TestTenderReviewState:
    def test_state_is_typed_dict(self):
        state: TenderReviewState = {
            "user_input": "hi",
            "request": {},
            "requirements": [],
            "retrieved_documents": [],
            "matched_response": [],
            "review_results": [],
        }
        assert state["user_input"] == "hi"
