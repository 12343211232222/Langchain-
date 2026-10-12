"""Multi-Agent Collaboration 核心数据模型（M10）。

AgentTask / AgentMessage / AgentResult / CollaborationPlan /
CollaborationResult。全部 dataclass，不引入任务队列框架。
"""

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from chatchat.agent.collaboration.exceptions import (
    InvalidMessageError,
    InvalidPlanError,
)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"  # 依赖失败，未执行


class MessageType(str, Enum):
    """最小消息类型集合，不设计几十种。"""

    TASK_REQUEST = "task_request"
    TASK_RESULT = "task_result"
    ERROR = "error"
    CONTEXT_UPDATE = "context_update"  # 预留


@dataclass
class AgentTask:
    """"请某一个 Agent 完成一个具体任务"。"""

    task_id: str
    agent_id: str
    task_type: str
    input: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    status: str = TaskStatus.PENDING.value

    def __post_init__(self):
        if not self.task_id or not self.agent_id or not self.task_type:
            raise InvalidPlanError(
                "AgentTask 的 task_id/agent_id/task_type 不能为空"
            )


@dataclass
class AgentMessage:
    """Agent 之间的结构化消息（不允许直接调用对方内部方法）。"""

    sender: str
    receiver: str
    message_type: str
    content: Any = None
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    correlation_id: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.sender or not self.receiver:
            raise InvalidMessageError("AgentMessage 必须有 sender 和 receiver")
        valid = {m.value for m in MessageType}
        if self.message_type not in valid:
            raise InvalidMessageError(f"未知 message_type: {self.message_type}")


@dataclass
class AgentResult:
    """一个 Agent 任务的统一结构化结果。"""

    task_id: str
    agent_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollaborationPlan:
    """Agent 协作流程定义。"""

    plan_id: str
    tasks: List[AgentTask]
    strategy: str = "sequential"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.plan_id:
            raise InvalidPlanError("plan_id 不能为空")
        if not self.tasks:
            raise InvalidPlanError("plan 至少包含一个 task")
        ids = [t.task_id for t in self.tasks]
        if len(ids) != len(set(ids)):
            raise InvalidPlanError("task_id 重复")
        known = set(ids)
        for task in self.tasks:
            for dep in task.dependencies:
                if dep not in known:
                    raise InvalidPlanError(
                        f"任务 {task.task_id} 依赖不存在的任务: {dep}"
                    )


@dataclass
class CollaborationResult:
    """一次多 Agent 协作的统一返回。"""

    success: bool
    plan_id: str = ""
    results: Dict[str, AgentResult] = field(default_factory=dict)
    failed_task: Optional[str] = None
    error: Optional[str] = None
    context: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
