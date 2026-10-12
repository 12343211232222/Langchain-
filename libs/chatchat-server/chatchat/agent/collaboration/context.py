"""CollaborationContext：协作层上下文（M10）。

与 M07 HarnessContext（Agent 执行上下文）严格隔离：
- CollaborationContext 只在协作层共享，不混成一个巨大 dict
- Agent 只通过 AgentTask.input + AgentMessage 获取自己任务所需数据
"""

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List

from chatchat.agent.collaboration.schemas import AgentMessage, AgentResult


@dataclass
class CollaborationContext:
    """多 Agent 协作共享上下文。"""

    user_input: str = ""
    collaboration_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    shared_data: Dict[str, Any] = field(default_factory=dict)
    messages: List[AgentMessage] = field(default_factory=list)
    task_results: Dict[str, AgentResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: AgentMessage) -> None:
        self.messages.append(message)

    def add_result(self, result: AgentResult) -> None:
        self.task_results[result.task_id] = result

    def set_shared(self, key: str, value: Any) -> None:
        self.shared_data[key] = value

    def get_shared(self, key: str, default: Any = None) -> Any:
        return self.shared_data.get(key, default)

    def get_dependency_outputs(self, dependencies: List[str]) -> Dict[str, Any]:
        """只取完成任务所需的上游输出（不无限复制全部上下文）。"""
        return {
            dep: self.task_results[dep].output
            for dep in dependencies
            if dep in self.task_results and self.task_results[dep].success
        }
