"""M10 Demo Agents（仅用于 Demo 与测试，非完整生产 Agent）。

RequirementAgent → RetrievalAgent → ComplianceAgent → ReportAgent

约定：
- 输入：Coordinator 发来的 JSON payload
  {"task_id","task_type","input","dependencies": {task_id: output_json}}
- 输出：HarnessResult(success=True, output=json.dumps(结构化结果))
- 无 llm 时使用确定性规则（保证 E2E Demo / 测试无真实 LLM）
"""

import json
import re
from typing import Any, Dict, Optional, Union

import json_repair

from chatchat.agent.harness import HarnessRequest, HarnessResult
from chatchat.agent.registry import (
    AgentDefinition,
    AgentRegistryInterface,
    CallableAgentFactory,
)
from chatchat.agent.registry.interfaces import AgentInterface
from chatchat.core.logging import get_logger

logger = get_logger(__name__)


def _parse_payload(request: Union[HarnessRequest, str]) -> Dict[str, Any]:
    raw = request if isinstance(request, str) else request.user_input
    parsed = json_repair.loads(raw)
    return parsed if isinstance(parsed, dict) else {"input": raw, "dependencies": {}}


def _dependency_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    """合并上游任务输出（JSON 字符串 → dict）。"""
    merged: Dict[str, Any] = {}
    for output in (payload.get("dependencies") or {}).values():
        parsed = json_repair.loads(output) if isinstance(output, str) else output
        if isinstance(parsed, dict):
            merged.update(parsed)
    return merged


def _ok(output: Dict[str, Any]) -> HarnessResult:
    return HarnessResult(success=True, output=json.dumps(output, ensure_ascii=False))


class RequirementAgent(AgentInterface):
    """从招标要求中提取结构化 Requirement。"""

    AGENT_ID = "requirement_agent"

    def __init__(self, llm=None):
        self._llm = llm

    @property
    def agent_id(self) -> str:
        return self.AGENT_ID

    def execute(self, request: Union[HarnessRequest, str]) -> HarnessResult:
        payload = _parse_payload(request)
        text = str(payload.get("input") or "")
        if self._llm is not None:
            from chatchat.agent.tender_review.prompts import (
                REQUIREMENT_EXTRACTION_PROMPT,
            )
            prompt = REQUIREMENT_EXTRACTION_PROMPT.format(tender_requirement=text)
            answer = self._llm.invoke([type("M", (), {"content": prompt})()])
            data = json_repair.loads(getattr(answer, "content", str(answer)))
            requirements = (data.get("requirements") or []) if isinstance(data, dict) else []
        else:
            # 确定性规则：按句切分（仅 Demo 用）
            sentences = [s.strip() for s in re.split(r"[。\n；;]", text) if s.strip()]
            requirements = [
                {"id": f"R{i + 1}", "text": s, "category": "other"}
                for i, s in enumerate(sentences)
            ]
        return _ok({"requirements": requirements})


class RetrievalAgent(AgentInterface):
    """按 Requirement 通过 M02 RAGService 检索证据（禁止直连向量库）。"""

    AGENT_ID = "retrieval_agent"

    def __init__(self, rag_service, knowledge_base: str = "", top_k: int = 5):
        self.rag_service = rag_service
        self.knowledge_base = knowledge_base
        self.top_k = top_k

    @property
    def agent_id(self) -> str:
        return self.AGENT_ID

    def execute(self, request: Union[HarnessRequest, str]) -> HarnessResult:
        from chatchat.rag.schemas import RAGQuery

        payload = _parse_payload(request)
        deps = _dependency_data(payload)
        requirements = deps.get("requirements") or []
        retrievals = []
        for req in requirements:
            evidences = []
            try:
                response = self.rag_service.retrieve(RAGQuery(
                    query=req.get("text", ""),
                    knowledge_base=self.knowledge_base,
                    top_k=self.top_k,
                ))
                evidences = [
                    {
                        "content": d.content,
                        "score": d.score,
                        "source": d.metadata.get("source"),
                        "page": d.metadata.get("page"),
                    }
                    for d in response.documents
                ]
            except Exception as e:
                logger.error(f"RetrievalAgent 检索失败({req.get('id')}): {e}")
            retrievals.append({
                "requirement_id": req.get("id"),
                "requirement": req.get("text"),
                "evidences": evidences,
            })
        return _ok({"retrievals": retrievals})


class ComplianceAgent(AgentInterface):
    """基于 Requirement + Evidence 输出 ComplianceResult。证据必须来自上游。"""

    AGENT_ID = "compliance_agent"

    def __init__(self, llm=None):
        self._llm = llm

    @property
    def agent_id(self) -> str:
        return self.AGENT_ID

    def execute(self, request: Union[HarnessRequest, str]) -> HarnessResult:
        payload = _parse_payload(request)
        deps = _dependency_data(payload)
        retrievals = deps.get("retrievals") or []
        results = []
        for item in retrievals:
            evidences = item.get("evidences") or []
            if self._llm is not None:
                from chatchat.agent.tender_review.prompts import (
                    COMPLIANCE_ANALYSIS_PROMPT,
                )
                prompt = COMPLIANCE_ANALYSIS_PROMPT.format(
                    requirement=item.get("requirement", ""),
                    matched_response="（由检索证据提供）",
                    evidence="\n".join(f"- {e['content']}" for e in evidences) or "（无）",
                )
                answer = self._llm.invoke([type("M", (), {"content": prompt})()])
                data = json_repair.loads(getattr(answer, "content", str(answer)))
                if not isinstance(data, dict):
                    data = {}
                status = data.get("status", "UNKNOWN")
                explanation = data.get("explanation", "")
            else:
                # 确定性规则（仅 Demo）：有证据 → COMPLIANT，无证据 → UNKNOWN
                if evidences:
                    status, explanation = "COMPLIANT", "存在检索证据支持该需求"
                else:
                    status, explanation = "UNKNOWN", "证据不足，无法判断"
            if not evidences:
                status = "UNKNOWN"  # 证据不足强制 UNKNOWN，不依赖 LLM 自觉
            results.append({
                "requirement_id": item.get("requirement_id"),
                "requirement": item.get("requirement"),
                "status": status,
                "explanation": explanation,
                "evidence": evidences,  # 直接来自 RetrievalAgent，不伪造
            })
        return _ok({"compliance_results": results})


class ReportAgent(AgentInterface):
    """汇总 ComplianceResult → TenderReviewReport。"""

    AGENT_ID = "report_agent"

    @property
    def agent_id(self) -> str:
        return self.AGENT_ID

    def execute(self, request: Union[HarnessRequest, str]) -> HarnessResult:
        payload = _parse_payload(request)
        deps = _dependency_data(payload)
        results = deps.get("compliance_results") or []
        summary = {
            "total": len(results),
            "compliant": sum(1 for r in results if r.get("status") == "COMPLIANT"),
            "partial": sum(1 for r in results if r.get("status") == "PARTIAL"),
            "non_compliant": sum(1 for r in results if r.get("status") == "NON_COMPLIANT"),
            "unknown": sum(1 for r in results if r.get("status") == "UNKNOWN"),
        }
        return _ok({"report": {"summary": summary, "results": results}})


_DEMO_DEFINITIONS = {
    RequirementAgent.AGENT_ID: AgentDefinition(
        agent_id=RequirementAgent.AGENT_ID,
        name="Requirement Agent (Demo)",
        description="M10 Demo：需求提取",
        tags=["demo", "m10"],
    ),
    RetrievalAgent.AGENT_ID: AgentDefinition(
        agent_id=RetrievalAgent.AGENT_ID,
        name="Retrieval Agent (Demo)",
        description="M10 Demo：证据检索",
        tags=["demo", "m10"],
    ),
    ComplianceAgent.AGENT_ID: AgentDefinition(
        agent_id=ComplianceAgent.AGENT_ID,
        name="Compliance Agent (Demo)",
        description="M10 Demo：符合性判断",
        tags=["demo", "m10"],
    ),
    ReportAgent.AGENT_ID: AgentDefinition(
        agent_id=ReportAgent.AGENT_ID,
        name="Report Agent (Demo)",
        description="M10 Demo：报告汇总",
        tags=["demo", "m10"],
    ),
}


def register_demo_agents(
    registry: AgentRegistryInterface,
    rag_service,
    llm=None,
    knowledge_base: str = "",
    top_k: int = 5,
) -> None:
    """注册 M10 四个 Demo Agent 到 M08 Registry。"""

    def build_requirement(definition, **kwargs):
        return RequirementAgent(llm=llm)

    def build_retrieval(definition, **kwargs):
        return RetrievalAgent(
            rag_service=rag_service, knowledge_base=knowledge_base, top_k=top_k
        )

    def build_compliance(definition, **kwargs):
        return ComplianceAgent(llm=llm)

    def build_report(definition, **kwargs):
        return ReportAgent()

    for agent_id, builder in [
        (RequirementAgent.AGENT_ID, build_requirement),
        (RetrievalAgent.AGENT_ID, build_retrieval),
        (ComplianceAgent.AGENT_ID, build_compliance),
        (ReportAgent.AGENT_ID, build_report),
    ]:
        if not registry.exists(agent_id):
            registry.register(
                _DEMO_DEFINITIONS[agent_id], CallableAgentFactory(builder)
            )
    logger.info("demo agents registered: requirement/retrieval/compliance/report")


def build_tender_review_plan(user_input: str, knowledge_base: str = "") -> "CollaborationPlan":
    """构建第一版 Tender Multi-Agent Review Plan（Sequential）。"""
    from chatchat.agent.collaboration.schemas import AgentTask, CollaborationPlan

    return CollaborationPlan(
        plan_id="tender_multi_agent_review",
        tasks=[
            AgentTask(
                task_id="t1_requirement",
                agent_id=RequirementAgent.AGENT_ID,
                task_type="extract_requirement",
                input=user_input,
            ),
            AgentTask(
                task_id="t2_retrieval",
                agent_id=RetrievalAgent.AGENT_ID,
                task_type="retrieve_evidence",
                dependencies=["t1_requirement"],
            ),
            AgentTask(
                task_id="t3_compliance",
                agent_id=ComplianceAgent.AGENT_ID,
                task_type="analyze_compliance",
                dependencies=["t2_retrieval"],
            ),
            AgentTask(
                task_id="t4_report",
                agent_id=ReportAgent.AGENT_ID,
                task_type="build_report",
                dependencies=["t3_compliance"],
            ),
        ],
        strategy="sequential",
    )
