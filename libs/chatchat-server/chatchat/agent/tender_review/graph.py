"""Tender Review Graph（M09）。

第一版线性 Graph：
Input → Extraction → Classification → Retrieval → Matching → Analysis → Aggregation

统一使用 M06 GraphBuilder（state_schema=TenderReviewState），
消除「GraphBuilder vs 裸 StateGraph」两套构图风格。
"""

from typing import Optional

from chatchat.agent.runtime.graph import GraphBuilder
from chatchat.agent.tender_review.nodes import (
    ComplianceAnalysisNode,
    RequirementClassificationNode,
    RequirementExtractionNode,
    ResponseMatchingNode,
    ResultAggregationNode,
    TenderInputNode,
    TenderRetrievalNode,
)
from chatchat.agent.tender_review.state import TenderReviewState


def build_tender_review_graph(
    rag_service,
    llm=None,
    knowledge_base: str = "",
    top_k: int = 5,
    retrieval_strategy: Optional[str] = None,
):
    """构建 Tender Review 编译图。依赖全部注入。"""
    return (
        GraphBuilder(state_schema=TenderReviewState)
        .add_node("input", TenderInputNode())
        .add_node("extract", RequirementExtractionNode(llm=llm))
        .add_node("classify", RequirementClassificationNode(llm=llm))
        .add_node(
            "retrieve",
            TenderRetrievalNode(
                rag_service=rag_service,
                knowledge_base=knowledge_base,
                top_k=top_k,
                retrieval_strategy=retrieval_strategy,
            ),
        )
        .add_node("match", ResponseMatchingNode(llm=llm))
        .add_node("analyze", ComplianceAnalysisNode(llm=llm))
        .add_node("aggregate", ResultAggregationNode())
        .set_entry("input")
        .add_edge("input", "extract")
        .add_edge("extract", "classify")
        .add_edge("classify", "retrieve")
        .add_edge("retrieve", "match")
        .add_edge("match", "analyze")
        .add_edge("analyze", "aggregate")
        .build()
    )
