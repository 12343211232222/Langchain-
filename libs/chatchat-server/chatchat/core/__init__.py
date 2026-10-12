"""ChatChat 核心基础层（Core）。

提供统一的异常体系与日志入口，作为后续 RAG Service、
LangGraph Runtime、Agent Harness 等模块的基础设施。

M01 阶段仅包含：
- ``exceptions``：项目级基础异常
- ``logging``：统一日志入口（复用现有 loguru 配置）
"""

from chatchat.core.exceptions import (
    ChatchatError,
    ConfigurationError,
    ServiceError,
    ResourceNotFoundError,
    ValidationError,
)
from chatchat.core.logging import get_logger

__all__ = [
    "ChatchatError",
    "ConfigurationError",
    "ServiceError",
    "ResourceNotFoundError",
    "ValidationError",
    "get_logger",
]