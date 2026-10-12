"""ChatChat Agent Registry 统一注册与发现层（M08）。

提供：
- ``schemas``：AgentDefinition / AgentRegistration / AgentStatus
- ``interfaces``：AgentInterface / AgentFactory / AgentRegistryInterface / HarnessAgent
- ``registry``：InMemoryAgentRegistry
- ``factory``：CallableAgentFactory
- ``exceptions``：AgentRegistryError 及子类（M01 体系）

M08 明确不做：业务 Agent（M09）、Multi-Agent（M10）、
动态插件系统、分布式 Registry、复杂生命周期平台。
"""

from chatchat.agent.registry.exceptions import (
    AgentAlreadyExistsError,
    AgentNotFoundError,
    AgentRegistryError,
    InvalidAgentDefinitionError,
    InvalidAgentFactoryError,
)
from chatchat.agent.registry.factory import CallableAgentFactory
from chatchat.agent.registry.interfaces import (
    AgentFactory,
    AgentInterface,
    AgentRegistryInterface,
    HarnessAgent,
)
from chatchat.agent.registry.registry import InMemoryAgentRegistry
from chatchat.agent.registry.schemas import (
    AgentDefinition,
    AgentRegistration,
    AgentStatus,
)

__all__ = [
    "AgentDefinition",
    "AgentRegistration",
    "AgentStatus",
    "AgentInterface",
    "HarnessAgent",
    "AgentFactory",
    "CallableAgentFactory",
    "AgentRegistryInterface",
    "InMemoryAgentRegistry",
    "AgentRegistryError",
    "AgentAlreadyExistsError",
    "AgentNotFoundError",
    "InvalidAgentDefinitionError",
    "InvalidAgentFactoryError",
]
