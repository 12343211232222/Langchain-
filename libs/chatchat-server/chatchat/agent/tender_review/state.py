"""Tender Review 业务 State（M09）。

扩展 M06 ``AgentState``（TypedDict 继承），保留 user_input/messages/
context/final_output/error/execution_status，增加业务字段。
不复制 M06 已有字段。
"""

from typing import Any, Dict, List, Optional, TypedDict

from chatchat.agent.runtime.state import AgentState


class TenderReviewState(AgentState, total=False):
    """Tender Review Graph 状态。

    字段（在 AgentState 基础上扩展）:
        request: 序列化后的 TenderReviewRequest（dict）。
        requirements: 提取出的需求列表（dict: id/text/category）。
        retrieved_documents: 按需求分组的检索结果
            [{"requirement_id": ..., "documents": [...]}]。
        matched_response: 按需求分组的匹配响应
            [{"requirement_id": ..., "text": str | None}]。
        review_results: 按需求分组的审查结果（dict 列表）。
    """

    request: Dict[str, Any]
    requirements: List[Dict[str, Any]]
    retrieved_documents: List[Any]
    matched_response: List[Dict[str, Any]]
    review_results: List[Dict[str, Any]]
