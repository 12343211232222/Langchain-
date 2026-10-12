"""进程内 Agent Registry 实现。

Dict[str, AgentRegistration]。不做自动扫描 / 动态加载 / 分布式，
未来需要时可升级为持久化 Registry。
"""

from typing import Dict, List

from chatchat.agent.registry.exceptions import (
    AgentAlreadyExistsError,
    AgentNotFoundError,
    InvalidAgentDefinitionError,
    InvalidAgentFactoryError,
)
from chatchat.agent.registry.interfaces import (
    AgentFactory,
    AgentInterface,
    AgentRegistryInterface,
)
from chatchat.agent.registry.schemas import (
    AgentDefinition,
    AgentRegistration,
    AgentStatus,
)
from chatchat.core.logging import get_logger

logger = get_logger(__name__)


class InMemoryAgentRegistry(AgentRegistryInterface):
    """默认进程内 Agent Registry。"""

    def __init__(self):
        self._registrations: Dict[str, AgentRegistration] = {}

    def register(self, definition: AgentDefinition, factory: AgentFactory) -> None:
        if not isinstance(definition, AgentDefinition):
            raise InvalidAgentDefinitionError(
                f"definition 必须是 AgentDefinition，得到 {type(definition).__name__}"
            )
        if not isinstance(factory, AgentFactory):
            raise InvalidAgentFactoryError(
                f"factory 必须是 AgentFactory，得到 {type(factory).__name__}"
            )
        agent_id = definition.agent_id
        if agent_id in self._registrations:
            raise AgentAlreadyExistsError(f"agent_id 已注册: {agent_id}")
        self._registrations[agent_id] = AgentRegistration(
            definition=definition, factory=factory, status=AgentStatus.REGISTERED
        )
        logger.info(f"agent registered: {agent_id} (v{definition.version})")

    def unregister(self, agent_id: str) -> None:
        if agent_id not in self._registrations:
            raise AgentNotFoundError(f"agent_id 不存在: {agent_id}")
        del self._registrations[agent_id]
        logger.info(f"agent unregistered: {agent_id}")

    def get(self, agent_id: str) -> AgentDefinition:
        registration = self._registrations.get(agent_id)
        if registration is None:
            raise AgentNotFoundError(f"agent_id 不存在: {agent_id}")
        return registration.definition

    def list(self) -> List[AgentDefinition]:
        return [r.definition for r in self._registrations.values()]

    def exists(self, agent_id: str) -> bool:
        return agent_id in self._registrations

    def create(self, agent_id: str, **kwargs) -> AgentInterface:
        registration = self._registrations.get(agent_id)
        if registration is None:
            raise AgentNotFoundError(f"agent_id 不存在: {agent_id}")
        if registration.status == AgentStatus.DISABLED:
            raise AgentNotFoundError(f"agent_id 已禁用: {agent_id}")
        agent = registration.factory.create(registration.definition, **kwargs)
        registration.status = AgentStatus.ENABLED
        return agent
