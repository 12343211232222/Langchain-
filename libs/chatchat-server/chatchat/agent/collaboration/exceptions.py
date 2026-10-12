"""Collaboration 层异常（M10）。

复用 M01 异常体系；Agent 不存在等错误直接复用 M08 的
``AgentNotFoundError``，不重复定义。
"""

from chatchat.core.exceptions import ChatchatError


class CollaborationError(ChatchatError):
    """Multi-Agent Collaboration 层统一异常。"""


class InvalidPlanError(CollaborationError):
    """CollaborationPlan 非法（空任务、依赖不存在、循环依赖等）。"""


class InvalidMessageError(CollaborationError):
    """AgentMessage 非法（缺 sender/receiver 等）。"""


class DependencyFailureError(CollaborationError):
    """上游任务失败导致下游任务不可执行。"""
