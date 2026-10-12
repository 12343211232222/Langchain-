"""Agent Registry 核心接口。

- ``AgentInterface``：最小统一 Agent 抽象（execute 委托 M07 Harness）
- ``AgentFactory``：根据 AgentDefinition 创建 Agent（"怎么创建"）
- ``AgentRegistryInterface``：注册/查找/列表/创建（"找到谁"，create 委托 Factory）

调用关系（规格二十）：
AgentDefinition → AgentRegistration → AgentRegistry → AgentFactory
→ Agent → AgentHarness → LangGraphRuntime
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Union

from chatchat.agent.harness import (
    AgentHarnessInterface,
    HarnessRequest,
    HarnessResult,
)
from chatchat.agent.registry.schemas import AgentDefinition


class AgentInterface(ABC):
    """最小统一 Agent 抽象。

    不重新实现 Agent Runtime；``execute`` 必须通过 M07 Harness。
    """

    @property
    @abstractmethod
    def agent_id(self) -> str:
        """Agent 唯一标识。"""
        raise NotImplementedError

    @abstractmethod
    def execute(self, request: Union[HarnessRequest, str]) -> HarnessResult:
        """执行一次 Agent 调用。"""
        raise NotImplementedError


class HarnessAgent(AgentInterface):
    """组合 M07 Harness 的通用 Agent 适配器（非业务 Agent）。

    把任意 ``AgentHarnessInterface`` 包装为 ``AgentInterface``，
    是 Registry → Harness → Runtime 接线关系的标准实现。
    """

    def __init__(
        self,
        definition: AgentDefinition,
        harness: AgentHarnessInterface,
    ):
        self._definition = definition
        self._harness = harness

    @property
    def agent_id(self) -> str:
        return self._definition.agent_id

    @property
    def definition(self) -> AgentDefinition:
        return self._definition

    def execute(self, request: Union[HarnessRequest, str]) -> HarnessResult:
        return self._harness.execute(request)


class AgentFactory(ABC):
    """Agent 工厂：根据 Definition 创建 Agent 实例。"""

    @abstractmethod
    def create(self, definition: AgentDefinition, **kwargs: Any) -> AgentInterface:
        """创建 Agent 实例。"""
        raise NotImplementedError


class AgentRegistryInterface(ABC):
    """Agent 注册与发现接口（进程内实现）。"""

    @abstractmethod
    def register(self, definition: AgentDefinition, factory: AgentFactory) -> None:
        """注册 Agent。重复 agent_id 抛 AgentAlreadyExistsError。"""
        raise NotImplementedError

    @abstractmethod
    def unregister(self, agent_id: str) -> None:
        """注销 Agent。不存在抛 AgentNotFoundError。"""
        raise NotImplementedError

    @abstractmethod
    def get(self, agent_id: str) -> AgentDefinition:
        """查询 Agent 元信息。不存在抛 AgentNotFoundError。"""
        raise NotImplementedError

    @abstractmethod
    def list(self) -> List[AgentDefinition]:
        """列出全部已注册 Agent 元信息。"""
        raise NotImplementedError

    @abstractmethod
    def exists(self, agent_id: str) -> bool:
        """判断 agent_id 是否已注册。"""
        raise NotImplementedError

    @abstractmethod
    def create(self, agent_id: str, **kwargs: Any) -> AgentInterface:
        """创建 Agent 实例（委托 Factory）。"""
        raise NotImplementedError
