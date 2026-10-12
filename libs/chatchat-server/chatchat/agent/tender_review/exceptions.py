"""Tender Review Agent 异常（M09）。

复用 M01 异常体系。
"""

from chatchat.core.exceptions import ChatchatError


class TenderReviewError(ChatchatError):
    """Tender Review Agent 业务异常。"""
