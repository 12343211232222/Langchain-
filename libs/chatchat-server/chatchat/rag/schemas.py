"""RAG 层统一数据结构。

定义 RAG 查询、文档与响应的统一 Schema，使上层（Chat / Agent / 未来 LangGraph）
不再依赖具体 VectorStore 的返回结构。

本阶段使用 dataclass 保持轻量，避免 pydantic 版本耦合。
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RAGQuery:
    """描述一次 RAG 检索请求。

    参数:
        query: 用户查询文本。
        knowledge_base: 指定知识库名称（可选，空串表示不限定）。
        top_k: 返回候选数量。
        score_threshold: 可选相似度阈值；None 表示使用项目默认值。
        filters: 为未来 metadata filtering 预留，M02 不实现复杂过滤。
        retrieval_strategy: 检索策略（M05 扩展）。None 表示默认 "vector"，
            可选值见 ``chatchat.rag.retrieval.RetrievalStrategy``。
    """

    query: str
    knowledge_base: str = ""
    top_k: int = 5
    score_threshold: Optional[float] = None
    filters: Optional[dict] = None
    retrieval_strategy: Optional[str] = None


@dataclass
class RAGDocument:
    """统一表示一个被检索到的文档/Chunk。

    只保存当前检索实现能提供的信息，不为未来 Citation 强行制造字段。
    """

    content: str
    score: Optional[float] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class RAGResponse:
    """一次 RAG Service 检索的统一返回。"""

    query: str
    documents: List[RAGDocument] = field(default_factory=list)
    retrieval_time: Optional[float] = None
    metadata: dict = field(default_factory=dict)