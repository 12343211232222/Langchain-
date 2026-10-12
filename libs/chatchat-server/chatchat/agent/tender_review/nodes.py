"""Tender Review 业务节点（M09）。

每个节点职责单一（extract / classify / retrieve / match / analyze /
aggregate），为 M10 复用做准备。LLM / RAGService 全部构造函数注入。
LLM 输出用项目已有依赖 ``json_repair`` 解析，不修改 M06 LLMNode。
"""

import json
from typing import Any, Dict, List, Optional

import json_repair
from langchain_core.messages import HumanMessage

from chatchat.agent.runtime.node import NodeBase
from chatchat.agent.tender_review.prompts import (
    COMPLIANCE_ANALYSIS_PROMPT,
    REQUIREMENT_CLASSIFICATION_PROMPT,
    REQUIREMENT_EXTRACTION_PROMPT,
)
from chatchat.agent.tender_review.schemas import (
    ComplianceStatus,
    Evidence,
    RequirementCategory,
    TenderReviewResult,
)
from chatchat.agent.tender_review.state import TenderReviewState
from chatchat.core.logging import get_logger

logger = get_logger(__name__)


def _parse_json(text: str) -> Dict[str, Any]:
    """解析 LLM JSON 输出；失败返回空 dict（节点自行降级为 UNKNOWN）。"""
    parsed = json_repair.loads(text)
    return parsed if isinstance(parsed, dict) else {}


def _default_llm():
    """延迟复用项目现有 LLM 创建方式（不新建第二套 LLM Client）。"""
    from chatchat.server.utils import get_ChatOpenAI

    return get_ChatOpenAI(streaming=False)


class _LLMNode(NodeBase):
    """持有可注入 LLM 的节点基类。"""

    def __init__(self, llm=None, name: Optional[str] = None):
        super().__init__(name)
        self._llm = llm

    def _ask(self, prompt: str) -> str:
        llm = self._llm or _default_llm()
        if self._llm is None:
            self._llm = llm
        message = llm.invoke([HumanMessage(content=prompt)])
        return getattr(message, "content", str(message))


class TenderInputNode(NodeBase):
    """输入节点：解析 user_input（JSON 请求或纯文本）为 request。"""

    def run(self, state: TenderReviewState) -> dict:
        raw = state.get("user_input", "")
        request: Dict[str, Any]
        parsed = json_repair.loads(raw)
        if isinstance(parsed, dict) and "tender_requirement" in parsed:
            request = parsed
        else:
            request = {"tender_requirement": raw, "bidder_response": ""}
        return {"request": request, "execution_status": "running"}


class RequirementExtractionNode(_LLMNode):
    """需求提取：LLM 结构化输出 requirements。"""

    def run(self, state: TenderReviewState) -> dict:
        request = state.get("request") or {}
        tender_text = request.get("tender_requirement", "")
        if not tender_text.strip():
            logger.warning("tender_requirement 为空，跳过提取")
            return {"requirements": []}
        prompt = REQUIREMENT_EXTRACTION_PROMPT.format(tender_requirement=tender_text)
        try:
            data = _parse_json(self._ask(prompt))
            if "requirements" not in data:
                raise ValueError("LLM 输出缺少 requirements 字段")
        except Exception as e:
            logger.error(f"需求提取失败: {e}")
            return {"requirements": [], "error": f"requirement_extraction: {e}"}
        requirements = [
            {"id": r.get("id", f"req_{i + 1}"), "text": r.get("text", ""),
             "category": RequirementCategory.OTHER.value}
            for i, r in enumerate(data.get("requirements") or [])
            if r.get("text")
        ]
        return {"requirements": requirements}


class RequirementClassificationNode(_LLMNode):
    """需求分类：LLM 结构化输出每条需求的 category。"""

    def run(self, state: TenderReviewState) -> dict:
        requirements = [dict(r) for r in (state.get("requirements") or [])]
        if not requirements:
            return {"requirements": []}
        prompt = REQUIREMENT_CLASSIFICATION_PROMPT.format(
            requirements=json.dumps(requirements, ensure_ascii=False)
        )
        try:
            data = _parse_json(self._ask(prompt))
            mapping = {
                c.get("id"): c.get("category")
                for c in (data.get("classifications") or [])
            }
        except Exception as e:
            logger.error(f"需求分类失败: {e}")
            mapping = {}
        valid = {c.value for c in RequirementCategory}
        for req in requirements:
            category = mapping.get(req["id"])
            req["category"] = category if category in valid else RequirementCategory.OTHER.value
        return {"requirements": requirements}


class TenderRetrievalNode(NodeBase):
    """RAG 检索：只通过 M02 RAGServiceInterface，禁止直连 Vector DB/BM25。"""

    def __init__(
        self,
        rag_service,
        knowledge_base: str = "",
        top_k: int = 5,
        retrieval_strategy: Optional[str] = None,
        name: Optional[str] = None,
    ):
        super().__init__(name)
        self.rag_service = rag_service
        self.knowledge_base = knowledge_base
        self.top_k = top_k
        self.retrieval_strategy = retrieval_strategy

    def run(self, state: TenderReviewState) -> dict:
        from chatchat.rag.schemas import RAGQuery

        request = state.get("request") or {}
        knowledge_base = request.get("knowledge_base") or self.knowledge_base
        grouped: List[Dict[str, Any]] = []
        for req in state.get("requirements") or []:
            documents: List[Dict[str, Any]] = []
            if knowledge_base:
                try:
                    response = self.rag_service.retrieve(RAGQuery(
                        query=req["text"],
                        knowledge_base=knowledge_base,
                        top_k=self.top_k,
                        retrieval_strategy=self.retrieval_strategy,
                    ))
                    documents = [
                        {
                            "content": d.content,
                            "score": d.score,
                            "source": d.metadata.get("source"),
                            "page": d.metadata.get("page"),
                        }
                        for d in response.documents
                    ]
                except Exception as e:
                    # RAG 异常不中断流程：该需求按证据不足处理
                    logger.error(f"RAG 检索失败(req={req['id']}): {e}")
            grouped.append({"requirement_id": req["id"], "documents": documents})
        return {"retrieved_documents": grouped}


class ResponseMatchingNode(_LLMNode):
    """响应匹配：从投标响应中为每条需求找出匹配内容；找不到为 None。"""

    def run(self, state: TenderReviewState) -> dict:
        request = state.get("request") or {}
        bidder_response = request.get("bidder_response", "")
        matched: List[Dict[str, Any]] = []
        for req in state.get("requirements") or []:
            text: Optional[str] = None
            if bidder_response.strip() and req["text"].strip() in bidder_response:
                text = req["text"]
            elif bidder_response.strip():
                # 简单子串未命中时，用 LLM 在响应中定位相关句；不编造
                prompt = (
                    "在下面的投标响应中找出与招标需求最相关的原句，"
                    "只输出原句，找不到输出 NONE。\n"
                    f"招标需求：{req['text']}\n投标响应：{bidder_response}"
                )
                try:
                    answer = self._ask(prompt).strip()
                    if answer and answer != "NONE" and answer in bidder_response:
                        text = answer
                except Exception as e:
                    logger.error(f"响应匹配失败(req={req['id']}): {e}")
            matched.append({"requirement_id": req["id"], "text": text})
        return {"matched_response": matched}


class ComplianceAnalysisNode(_LLMNode):
    """符合性分析：LLM 判断 status/explanation/risk；证据来自真实检索结果。"""

    def run(self, state: TenderReviewState) -> dict:
        requirements = state.get("requirements") or []
        docs_map = {
            g["requirement_id"]: g["documents"]
            for g in (state.get("retrieved_documents") or [])
        }
        match_map = {
            m["requirement_id"]: m["text"]
            for m in (state.get("matched_response") or [])
        }
        results: List[Dict[str, Any]] = []
        for req in requirements:
            matched_text = match_map.get(req["id"])
            docs = docs_map.get(req["id"], [])
            evidence = [
                Evidence(content=d["content"], source=d.get("source"),
                         page=d.get("page"), score=d.get("score"))
                for d in docs
            ]
            evidence_text = "\n".join(
                f"- {d['content']}" for d in docs
            ) or "（无检索证据）"
            prompt = COMPLIANCE_ANALYSIS_PROMPT.format(
                requirement=req["text"],
                matched_response=matched_text or "（未找到匹配的投标响应）",
                evidence=evidence_text,
            )
            try:
                data = _parse_json(self._ask(prompt))
                status = data.get("status", ComplianceStatus.UNKNOWN.value)
                if status not in {s.value for s in ComplianceStatus}:
                    status = ComplianceStatus.UNKNOWN.value
                explanation = data.get("explanation", "")
                risk = data.get("risk", "")
            except Exception as e:
                # LLM 解析/调用失败：降级 UNKNOWN，不强行 COMPLIANT
                logger.error(f"符合性分析失败(req={req['id']}): {e}")
                status = ComplianceStatus.UNKNOWN.value
                explanation = f"分析失败: {e}"
                risk = ""
            if not docs and matched_text is None:
                # 证据不足：强制 UNKNOWN（不依赖 LLM 自觉）
                status = ComplianceStatus.UNKNOWN.value
                explanation = explanation or "证据不足，无法判断"
            results.append({
                "requirement": req["text"],
                "category": req.get("category", RequirementCategory.OTHER.value),
                "status": status,
                "matched_response": matched_text,
                "evidence": [e.__dict__ for e in evidence],
                "explanation": explanation,
                "risk": risk,
            })
        return {"review_results": results}


class ResultAggregationNode(NodeBase):
    """结果汇总：统计各状态数量，输出结构化 TenderReviewResult。"""

    def run(self, state: TenderReviewState) -> dict:
        results = state.get("review_results") or []
        counts = {s.value: 0 for s in ComplianceStatus}
        for r in results:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        result = TenderReviewResult(
            results=[],  # 序列化时不重复内联 results 对象，下面统一走 dict
            total_requirements=len(results),
            compliant_count=counts[ComplianceStatus.COMPLIANT.value],
            partial_count=counts[ComplianceStatus.PARTIAL.value],
            non_compliant_count=counts[ComplianceStatus.NON_COMPLIANT.value],
            unknown_count=counts[ComplianceStatus.UNKNOWN.value],
        )
        final = result.to_dict()
        final["results"] = results
        return {
            "final_output": json.dumps(final, ensure_ascii=False),
            "execution_status": "success",
        }
