"""Agent Registry 数据模型。

``AgentDefinition`` 只描述 Agent 元信息，不是业务 Agent 实例；
``AgentRegistration`` = Definition + Factory + 生命周期状态。
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from chatchat.agent.registry.exceptions import InvalidAgentDefinitionError


class AgentStatus(str, Enum):
    """最小生命周期状态（不做心跳/调度/健康检查平台）。"""

    REGISTERED = "registered"
    ENABLED = "enabled"
    DISABLED = "disabled"


@dataclass
class AgentDefinition:
    """Agent 元信息。

    参数:
        agent_id: 唯一标识（必填，非空）。
        name: 展示名。
        description: 描述。
        version: 版本字符串（仅保存，不做 SemVer 路由/灰度/回滚）。
        metadata: 业务元信息（Registry 只保存与查询，不解释）。
        tags: 标签。
        enabled: 是否启用。
    """

    agent_id: str
    name: str = ""
    description: str = ""
    version: str = "0.1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    enabled: bool = True

    def __post_init__(self):
        if not isinstance(self.agent_id, str) or not self.agent_id.strip():
            raise InvalidAgentDefinitionError("agent_id 必须是非空字符串")
        if self.metadata is None or not isinstance(self.metadata, dict):
            raise InvalidAgentDefinitionError("metadata 必须是 dict")


@dataclass
class AgentRegistration:
    """Registry 内部保存的注册项：Definition + Factory + 状态。"""

    definition: AgentDefinition
    factory: Any  # AgentFactory（避免循环 import，这里用 Any 并在 register 时校验）
    status: AgentStatus = AgentStatus.REGISTERED
