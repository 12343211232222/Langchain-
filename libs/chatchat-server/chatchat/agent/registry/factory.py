"""Agent Factory 实现。

Registry 负责"找到谁"，Factory 负责"怎么创建"。
"""

from typing import Any, Callable

from chatchat.agent.registry.interfaces import AgentFactory, AgentInterface
from chatchat.agent.registry.schemas import AgentDefinition


class CallableAgentFactory(AgentFactory):
    """把普通函数包装为 AgentFactory。

    函数签名：``fn(definition: AgentDefinition, **kwargs) -> AgentInterface``
    """

    def __init__(self, fn: Callable[..., AgentInterface]):
        self._fn = fn

    def create(self, definition: AgentDefinition, **kwargs: Any) -> AgentInterface:
        return self._fn(definition, **kwargs)
