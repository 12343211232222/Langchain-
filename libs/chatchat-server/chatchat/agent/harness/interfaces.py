"""Harness 层核心接口。

- ``AgentHarnessInterface``：统一执行入口
- Manager ABC 在各自模块中定义（context/memory/tools/policy），
  此处仅保留 Harness 主接口，避免重复。
"""

from abc import ABC, abstractmethod
from typing import Union

from chatchat.agent.harness.context import HarnessRequest
from chatchat.agent.harness.result import HarnessResult


class AgentHarnessInterface(ABC):
    """Agent Harness 抽象接口（位于 M06 Runtime 上层）。"""

    @abstractmethod
    def execute(self, request: Union[HarnessRequest, str]) -> HarnessResult:
        """执行一次 Agent 调用，返回统一 ``HarnessResult``。"""
        raise NotImplementedError
