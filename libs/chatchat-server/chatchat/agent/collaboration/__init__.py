"""Multi-Agent Collaboration Layer（M10）。

建立轻量级多 Agent 协作层：
- ``schemas``：AgentTask / AgentMessage / AgentResult / CollaborationPlan / CollaborationResult
- ``context``：CollaborationContext（与 M07 HarnessContext 隔离）
- ``strategies``：SequentialStrategy（拓扑依赖）/ ParallelStrategy（接口）
- ``coordinator``：AgentCoordinator（只依赖 M08 AgentRegistryInterface）
- ``demo_agents``：Requirement / Retrieval / Compliance / Report Demo Agents

M10 不做：Supervisor / Planner / Debate / Reflection / 分布式调度。
"""

from chatchat.agent.collaboration.context import CollaborationContext
from chatchat.agent.collaboration.coordinator import AgentCoordinator
from chatchat.agent.collaboration.demo_agents import (
    ComplianceAgent,
    ReportAgent,
    RequirementAgent,
    RetrievalAgent,
    build_tender_review_plan,
    register_demo_agents,
)
from chatchat.agent.collaboration.exceptions import (
    CollaborationError,
    DependencyFailureError,
    InvalidMessageError,
    InvalidPlanError,
)
from chatchat.agent.collaboration.schemas import (
    AgentMessage,
    AgentResult,
    AgentTask,
    CollaborationPlan,
    CollaborationResult,
    MessageType,
    TaskStatus,
)
from chatchat.agent.collaboration.strategies import (
    CollaborationStrategy,
    ParallelStrategy,
    SequentialStrategy,
)

__all__ = [
    "AgentTask",
    "AgentMessage",
    "AgentResult",
    "CollaborationContext",
    "CollaborationPlan",
    "CollaborationResult",
    "MessageType",
    "TaskStatus",
    "AgentCoordinator",
    "CollaborationStrategy",
    "SequentialStrategy",
    "ParallelStrategy",
    "RequirementAgent",
    "RetrievalAgent",
    "ComplianceAgent",
    "ReportAgent",
    "register_demo_agents",
    "build_tender_review_plan",
    "CollaborationError",
    "InvalidPlanError",
    "InvalidMessageError",
    "DependencyFailureError",
]
