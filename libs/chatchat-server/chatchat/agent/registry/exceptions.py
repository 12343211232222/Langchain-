"""Agent Registry 层异常。

复用 M01 异常体系（``ChatchatError``），不创建第二套基础异常。
"""

from chatchat.core.exceptions import ChatchatError


class AgentRegistryError(ChatchatError):
    """Agent Registry 层统一异常。"""


class AgentAlreadyExistsError(AgentRegistryError):
    """重复注册同一 agent_id。"""


class AgentNotFoundError(AgentRegistryError):
    """查询/创建不存在的 agent_id。"""


class InvalidAgentDefinitionError(AgentRegistryError):
    """AgentDefinition 非法（空 id、缺字段等）。"""


class InvalidAgentFactoryError(AgentRegistryError):
    """Factory 非法（缺少 create 方法等）。"""
