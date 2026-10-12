"""多路召回融合策略。

RRF：基于 rank 融合，不直接平均不同 Retriever 的原始 score。
Weighted：基于归一化 score 的加权和。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

from chatchat.core.exceptions import ValidationError
from chatchat.rag.retrieval.schemas import RetrievalResult


def _dedup_key(r: RetrievalResult) -> Tuple[str, str]:
    """同一文档在不同召回器中的去重标识。"""
    return (r.content, str(r.metadata.get("source", "")))


class FusionStrategy(ABC):
    """融合策略接口：将多路召回结果融合为统一排序列表。"""

    @abstractmethod
    def fuse(self, results: List[List[RetrievalResult]]) -> List[RetrievalResult]:
        raise NotImplementedError


class RRFFusion(FusionStrategy):
    """Reciprocal Rank Fusion：RRF(d) = Σ 1 / (k + rank_i(d))，仅依赖 rank。"""

    def __init__(self, k: int = 60):
        self.k = k

    def fuse(self, results: List[List[RetrievalResult]]) -> List[RetrievalResult]:
        fused: Dict[Tuple[str, str], RetrievalResult] = {}
        scores: Dict[Tuple[str, str], float] = {}
        for one in results:
            for r in one:
                key = _dedup_key(r)
                scores[key] = scores.get(key, 0.0) + 1.0 / (self.k + r.rank)
                if key not in fused:
                    fused[key] = RetrievalResult(
                        content=r.content,
                        metadata=dict(r.metadata),
                        retriever="fusion",
                    )
        ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        out = []
        for i, key in enumerate(ranked_keys):
            r = fused[key]
            r.score = scores[key]
            r.rank = i
            out.append(r)
        return out


class WeightedFusion(FusionStrategy):
    """加权融合：score(d) = Σ w_i × normalized_score_i(d)。

    不同 Retriever 的 score 分布不同，先对每路结果做 min-max 归一化再加权；
    缺失 score（None）按 0.0 处理。权重不允许为负，权重和必须大于 0，
    且自动归一化为和为 1。
    """

    def __init__(self, weights: List[float] = None):
        weights = list(weights) if weights is not None else [0.7, 0.3]
        if not weights or any(w < 0 for w in weights):
            raise ValidationError(f"非法融合权重: {weights}")
        total = sum(weights)
        if total <= 0:
            raise ValidationError(f"融合权重之和必须大于 0: {weights}")
        self.weights = [w / total for w in weights]

    def fuse(self, results: List[List[RetrievalResult]]) -> List[RetrievalResult]:
        if len(results) != len(self.weights):
            raise ValidationError(
                f"召回路数({len(results)})与权重数({len(self.weights)})不一致"
            )

        # 每路 min-max 归一化
        norm_scores: List[Dict[Tuple[str, str], float]] = []
        for one in results:
            raw = {id(r): (r.score if r.score is not None else 0.0) for r in one}
            values = list(raw.values())
            lo, hi = (min(values), max(values)) if values else (0.0, 0.0)
            span = hi - lo
            norm_scores.append(
                {rid: ((v - lo) / span if span > 0 else (1.0 if hi > 0 else 0.0))
                 for rid, v in raw.items()}
            )

        fused: Dict[Tuple[str, str], RetrievalResult] = {}
        scores: Dict[Tuple[str, str], float] = {}
        for one, weight, norm in zip(results, self.weights, norm_scores):
            for r in one:
                key = _dedup_key(r)
                scores[key] = scores.get(key, 0.0) + weight * norm.get(id(r), 0.0)
                if key not in fused:
                    fused[key] = RetrievalResult(
                        content=r.content,
                        metadata=dict(r.metadata),
                        retriever="fusion",
                    )
        ranked_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        out = []
        for i, key in enumerate(ranked_keys):
            r = fused[key]
            r.score = scores[key]
            r.rank = i
            out.append(r)
        return out