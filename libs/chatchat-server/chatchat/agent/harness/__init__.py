"""ChatChat Agent Harness 统一运行控制层（M07）。

位于 M06 LangGraph Runtime 上层，提供：
- ``context``：HarnessContext / ContextManager
- ``memory``：MemoryManager（InMemory 实现）
- ``tools``：ToolManager（适配既有 tools_registry）
- ``retry``：RetryPolicy
- ``policy``：ExecutionPolicy / AllowAllPolicy
- ``execution``：ExecutionManager（生命周期 + Retry）
- ``harness``：DefaultAgentHarness（统一入口）
- ``result``：HarnessResult
- ``exceptions``：HarnessError / PolicyViolationError

M07 明确不做：Agent Registry（M08）、业务 Agent（M09）、
Multi-Agent、复杂 Checkpoint/Streaming/长期 Memory。
"""

from chatchat.agent.harness.context import (
    ContextManager,
    DefaultContextManager,
    HarnessContext,
    HarnessRequest,
)
from chatchat.agent.harness.exceptions import HarnessError, PolicyViolationError
from chatchat.agent.harness.execution import ExecutionManager, ExecutionMetadata
from chatchat.agent.harness.harness import DefaultAgentHarness
from chatchat.agent.harness.interfaces import AgentHarnessInterface
from chatchat.agent.harness.memory import InMemoryMemoryManager, MemoryManager
from chatchat.agent.harness.policy import AllowAllPolicy, ExecutionPolicy
from chatchat.agent.harness.result import HarnessResult
from chatchat.agent.harness.retry import RetryPolicy
from chatchat.agent.harness.tools import RegistryToolManager, ToolManager

__all__ = [
    "HarnessRequest",
    "HarnessContext",
    "ContextManager",
    "DefaultContextManager",
    "MemoryManager",
    "InMemoryMemoryManager",
    "ToolManager",
    "RegistryToolManager",
    "RetryPolicy",
    "ExecutionPolicy",
    "AllowAllPolicy",
    "ExecutionManager",
    "ExecutionMetadata",
    "AgentHarnessInterface",
    "DefaultAgentHarness",
    "HarnessResult",
    "HarnessError",
    "PolicyViolationError",
]
