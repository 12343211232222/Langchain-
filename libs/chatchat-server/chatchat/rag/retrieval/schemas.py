"""Retrieval 层统一数据结构。

复用并扩展 M02 的 ``RAGDocument``，为检索结果补充召回来源与排名信息，
不重复定义高度相似的数据结构。
"""

from dataclasses import dataclass

from chatchat.rag.schemas import RAGDocument


@dataclass
class RetrievalResult(RAGDocument):
    """一路召回（vector / bm25 / fusion）产生的统一检索结果。

    参数:
        retriever: 召回来源标记，如 ``"vector"`` / ``"bm25"`` / ``"fusion"``。
        rank: 该结果在对应召回器中的排名（0 起）。
    """

    retriever: str = ""
    rank: int = 0