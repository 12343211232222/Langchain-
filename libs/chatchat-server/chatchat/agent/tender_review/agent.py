"""TenderReviewAgent（M09）与 M08 Registry 集成。

调用链：
AgentRegistry.create("tender_review")
  → TenderReviewAgent.execute(request)
  → AgentHarness.execute（M07：Retry/Policy/Context/Memory）
  → LangGraphRuntime.run（M06）
  → Tender Review Graph
  → TenderReviewResult

依赖全部构造函数注入；不在 Agent 内部创建全局单例；
不自行实现 Retry（由 M07 Harness 负责）。
"""

import json
from dataclasses import asdict
from typing import Optional, Union

from chatchat.agent.harness import (
    AgentHarnessInterface,
    DefaultAgentHarness,
    HarnessRequest,
    HarnessResult,
)
from chatchat.agent.registry import (
    AgentDefinition,
    AgentRegistryInterface,
    CallableAgentFactory,
)
from chatchat.agent.registry.interfaces import AgentInterface
from chatchat.agent.runtime import LangGraphRuntime
from chatchat.agent.tender_review.exceptions import TenderReviewError
from chatchat.agent.tender_review.schemas import (
    TenderReviewRequest,
    TenderReviewResult,
)
from chatchat.core.logging import get_logger

logger = get_logger(__name__)

TENDER_REVIEW_AGENT_ID = "tender_review"

TENDER_REVIEW_DEFINITION = AgentDefinition(
    agent_id=TENDER_REVIEW_AGENT_ID,
    name="Tender Review Agent",
    description="招标要求与投标响应符合性审查 Agent",
    version="0.1.0",
    metadata={"domain": "tender", "type": "business"},
    tags=["tender", "review", "compliance"],
)


class TenderReviewAgent(AgentInterface):
    """标书审查业务 Agent。

    参数:
        rag_service: M02 ``RAGServiceInterface``（必需）。
        harness: M07 ``AgentHarnessInterface``；为 None 时用内部 Graph
            组装 LangGraphRuntime + DefaultAgentHarness。
        llm: 可注入 LLM（测试用 Fake）；为 None 时各节点延迟使用
            项目现有 ``get_ChatOpenAI``。
        knowledge_base / top_k / retrieval_strategy: 透传给 Graph。
    """

    def __init__(
        self,
        rag_service,
        harness: Optional[AgentHarnessInterface] = None,
        llm=None,
        knowledge_base: str = "",
        top_k: int = 5,
        retrieval_strategy: Optional[str] = None,
    ):
        if rag_service is None:
            raise TenderReviewError("rag_service 必须注入（M02 RAGServiceInterface）")
        self.rag_service = rag_service
        self._definition = TENDER_REVIEW_DEFINITION
        if harness is not None:
            self._harness = harness
        else:
            from chatchat.agent.tender_review.graph import build_tender_review_graph

            graph = build_tender_review_graph(
                rag_service=rag_service,
                llm=llm,
                knowledge_base=knowledge_base,
                top_k=top_k,
                retrieval_strategy=retrieval_strategy,
            )
            self._harness = DefaultAgentHarness(LangGraphRuntime(graph))

    @property
    def agent_id(self) -> str:
        return self._definition.agent_id

    @property
    def definition(self) -> AgentDefinition:
        return self._definition

    def execute(
        self, request: Union[TenderReviewRequest, HarnessRequest, str]
    ) -> HarnessResult:
        """执行审查。业务输入为 ``TenderReviewRequest``。"""
        if isinstance(request, TenderReviewRequest):
            harness_request = HarnessRequest(
                user_input=json.dumps(asdict(request), ensure_ascii=False),
                session_id=request.session_id,
                metadata=dict(request.metadata),
            )
        else:
            harness_request = request
        return self._harness.execute(harness_request)

    def review(self, request: TenderReviewRequest) -> TenderReviewResult:
        """业务化入口：返回结构化 ``TenderReviewResult``。"""
        result = self.execute(request)
        if not result.success or not result.output:
            raise TenderReviewError(f"审查失败: {result.error or '无输出'}")
        return TenderReviewResult.from_dict(json.loads(result.output))


def register_tender_review_agent(
    registry: AgentRegistryInterface,
    rag_service,
    llm=None,
    **graph_defaults,
) -> None:
    """把 TenderReviewAgent 注册到 M08 Registry。

    之后通过 ``registry.create("tender_review")`` 获取实例。
    """

    def build(definition: AgentDefinition, **kwargs) -> TenderReviewAgent:
        merged = {**graph_defaults, **kwargs}
        return TenderReviewAgent(rag_service=rag_service, llm=llm, **merged)

    registry.register(TENDER_REVIEW_DEFINITION, CallableAgentFactory(build))
    logger.info(f"registered agent: {TENDER_REVIEW_AGENT_ID}")
