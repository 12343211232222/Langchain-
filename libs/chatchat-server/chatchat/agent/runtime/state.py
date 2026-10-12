"""统一 Agent State 抽象。

使用 TypedDict（``total=False``）定义图执行期间在节点间流转的状态。
节点返回的部分 dict 会被 LangGraph 合并进 State。

字段均为可选：不同图只需使用自己关心的子集。
"""

from typing import Any, Dict, List, TypedDict


class AgentState(TypedDict, total=False):
    """Runtime 统一状态。

    字段:
        user_input: 用户原始输入。
        messages: 对话消息列表（langchain Message 或 dict）。
        context: 自由上下文（Loop 计数、中间变量等）。
        retrieved_documents: RAGNode 检索结果（``RAGDocument`` 列表）。
        tool_results: 工具执行结果（M06 仅预留字段）。
        final_output: 最终输出文本。
        execution_status: 执行状态（pending/running/success/failed）。
        error: 失败原因描述。
    """

    user_input: str
    messages: List[Any]
    context: Dict[str, Any]
    retrieved_documents: List[Any]
    tool_results: List[Any]
    final_output: str
    execution_status: str
    error: str
