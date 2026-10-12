"""Tender Review Agent 数据模型（M09）。

输入：``TenderReviewRequest``
输出：``TenderReviewResult``（结构化审查结果，含证据引用）
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class RequirementCategory(str, Enum):
    """需求分类（保持最小集合，不过度设计）。"""

    QUALIFICATION = "qualification"
    TECHNICAL = "technical"
    EXPERIENCE = "experience"
    SERVICE = "service"
    COMMERCIAL = "commercial"
    OTHER = "other"


class ComplianceStatus(str, Enum):
    """符合性状态。证据不足时必须允许 UNKNOWN。"""

    COMPLIANT = "COMPLIANT"
    NON_COMPLIANT = "NON_COMPLIANT"
    PARTIAL = "PARTIAL"
    UNKNOWN = "UNKNOWN"


@dataclass
class TenderReviewRequest:
    """一次标书审查请求。

    参数:
        tender_requirement: 招标文件 / 招标要求文本。
        bidder_response: 投标响应文件文本。
        session_id: 会话标识。
        knowledge_base: RAG 检索的知识库（可选）。
        metadata: 附加信息。
    """

    tender_requirement: str
    bidder_response: str = ""
    session_id: str = ""
    knowledge_base: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Requirement:
    """一条可审查的招标需求。"""

    id: str
    text: str
    category: str = RequirementCategory.OTHER.value


@dataclass
class Evidence:
    """证据引用。source/page 必须来自真实 RAG metadata，允许 None，禁止伪造。"""

    content: str
    source: Optional[str] = None
    page: Optional[int] = None
    score: Optional[float] = None


@dataclass
class RequirementReviewResult:
    """单条需求的审查结果。"""

    requirement: str
    category: str = RequirementCategory.OTHER.value
    status: str = ComplianceStatus.UNKNOWN.value
    matched_response: Optional[str] = None
    evidence: List[Evidence] = field(default_factory=list)
    explanation: str = ""
    risk: str = ""


@dataclass
class TenderReviewResult:
    """一次标书审查的结构化结果。"""

    results: List[RequirementReviewResult] = field(default_factory=list)
    total_requirements: int = 0
    compliant_count: int = 0
    partial_count: int = 0
    non_compliant_count: int = 0
    unknown_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TenderReviewResult":
        results = [
            RequirementReviewResult(
                requirement=r.get("requirement", ""),
                category=r.get("category", RequirementCategory.OTHER.value),
                status=r.get("status", ComplianceStatus.UNKNOWN.value),
                matched_response=r.get("matched_response"),
                evidence=[Evidence(**e) for e in r.get("evidence", [])],
                explanation=r.get("explanation", ""),
                risk=r.get("risk", ""),
            )
            for r in data.get("results", [])
        ]
        return cls(
            results=results,
            total_requirements=data.get("total_requirements", len(results)),
            compliant_count=data.get("compliant_count", 0),
            partial_count=data.get("partial_count", 0),
            non_compliant_count=data.get("non_compliant_count", 0),
            unknown_count=data.get("unknown_count", 0),
        )
