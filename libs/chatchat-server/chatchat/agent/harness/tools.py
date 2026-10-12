"""Tool Manager：统一 Tool 管理接口。

适配现有 ``chatchat.server.agent.tools_factory`` 的 ``_TOOLS_REGISTRY``。
通过延迟 import 复用既有工具注册与调用逻辑，不重写已有工具。

支持：
1. register（额外注册 Harness 侧工具，不影响既有注册）
2. get / list（读取全部工具，含既有注册）
3. execute（调用工具）
4. load / save / clear（空操作接口，保持可扩展性）
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from chatchat.core.logging import get_logger

logger = get_logger(__name__)


class ToolManager(ABC):
    """工具生命周期接口。"""

    @abstractmethod
    def register(self, name: str, tool: Any) -> None:
        """注册一个工具（Harness 侧视图）。"""
        raise NotImplementedError

    @abstractmethod
    def get(self, name: str) -> Optional[Any]:
        """按名称获取工具。"""
        raise NotImplementedError

    @abstractmethod
    def list_tools(self) -> List[str]:
        """返回已注册工具名称列表。"""
        raise NotImplementedError

    @abstractmethod
    def execute(self, name: str, **kwargs: Any) -> Any:
        """执行指定工具并返回结果。"""
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        """加载外部工具配置（当前阶段空实现，保留扩展点）。"""
        raise NotImplementedError

    @abstractmethod
    def save(self) -> None:
        """保存工具状态（当前阶段空实现，保留扩展点）。"""
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """清空 Harness 侧注册（不影响既有全局注册）。"""
        raise NotImplementedError


class RegistryToolManager(ToolManager):
    """复用既有 ``_TOOLS_REGISTRY`` 的 ToolManager。

    延迟导入 ``tools_registry`` 与 ``_TOOLS_REGISTRY``，避免：
    1. 循环依赖
    2. 启动时即加载全部工具重资源
    """

    def __init__(self):
        self._tools: Dict[str, Any] = {}

    def _registry(self):
        """延迟加载既有全局注册表。"""
        from chatchat.server.agent.tools_factory.tools_registry import (
            _TOOLS_REGISTRY,
        )
        return _TOOLS_REGISTRY

    def register(self, name: str, tool: Any) -> None:
        """在 Harness 侧独立注册（不污染既有全局注册表）。"""
        self._tools[name] = tool
        logger.info(f"tool registered: {name}")

    def get(self, name: str) -> Optional[Any]:
        """先查 Harness 侧，再查既有全局注册表。"""
        if name in self._tools:
            return self._tools[name]
        try:
            return self._registry().get(name)
        except Exception:
            return None

    def list_tools(self) -> List[str]:
        """返回并集（Harness 侧优先，不重复）。"""
        names = set(self._tools)
        try:
            names |= set(self._registry().keys())
        except Exception:
            pass
        return sorted(names)

    def execute(self, name: str, **kwargs: Any) -> Any:
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"工具未找到: {name}")
        logger.info(f"tool execute: {name}")
        if callable(tool):
            return tool(**kwargs)
        return tool.run(**kwargs)

    def load(self) -> None:
        pass

    def save(self) -> None:
        pass

    def clear(self) -> None:
        self._tools.clear()
