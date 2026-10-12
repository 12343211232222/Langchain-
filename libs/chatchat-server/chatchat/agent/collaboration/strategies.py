"""协作策略（M10）。

- ``SequentialStrategy``：按 dependencies 做拓扑排序执行（不是 list 顺序）
- ``ParallelStrategy``：第一版只定义接口（分组），不实现分布式调度
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Set

from chatchat.agent.collaboration.exceptions import InvalidPlanError
from chatchat.agent.collaboration.schemas import AgentTask, CollaborationPlan


def topological_order(tasks: List[AgentTask]) -> List[AgentTask]:
    """Kahn 拓扑排序。循环依赖抛 InvalidPlanError。"""
    by_id: Dict[str, AgentTask] = {t.task_id: t for t in tasks}
    indegree: Dict[str, int] = {t.task_id: len(t.dependencies) for t in tasks}
    dependents: Dict[str, List[str]] = {t.task_id: [] for t in tasks}
    for t in tasks:
        for dep in t.dependencies:
            dependents[dep].append(t.task_id)

    ready = [tid for tid, d in indegree.items() if d == 0]
    order: List[AgentTask] = []
    while ready:
        tid = ready.pop(0)
        order.append(by_id[tid])
        for nxt in dependents[tid]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
    if len(order) != len(tasks):
        raise InvalidPlanError("plan 存在循环依赖")
    return order


def topological_levels(tasks: List[AgentTask]) -> List[List[AgentTask]]:
    """按依赖层级分组（同层任务互相无依赖，可并行）。"""
    by_id: Dict[str, AgentTask] = {t.task_id: t for t in tasks}
    level: Dict[str, int] = {}

    def resolve(tid: str, visiting: Set[str]) -> int:
        if tid in level:
            return level[tid]
        if tid in visiting:
            raise InvalidPlanError("plan 存在循环依赖")
        visiting.add(tid)
        task = by_id[tid]
        lv = 0 if not task.dependencies else 1 + max(
            resolve(d, visiting) for d in task.dependencies
        )
        visiting.discard(tid)
        level[tid] = lv
        return lv

    for t in tasks:
        resolve(t.task_id, set())
    groups: Dict[int, List[AgentTask]] = {}
    for tid, lv in level.items():
        groups.setdefault(lv, []).append(by_id[tid])
    return [groups[i] for i in sorted(groups)]


class CollaborationStrategy(ABC):
    """协作执行策略接口。"""

    @abstractmethod
    def order(self, plan: CollaborationPlan) -> List[AgentTask]:
        """返回任务执行顺序。"""
        raise NotImplementedError


class SequentialStrategy(CollaborationStrategy):
    """按 dependencies 拓扑顺序执行（A→B→C）。"""

    def order(self, plan: CollaborationPlan) -> List[AgentTask]:
        return topological_order(plan.tasks)


class ParallelStrategy(CollaborationStrategy):
    """并行策略（第一版仅接口）。

    实现时只允许 Python 标准库 concurrent.futures；
    不引入 Celery / Kafka / Redis / RabbitMQ。
    """

    def order(self, plan: CollaborationPlan) -> List[AgentTask]:
        return topological_order(plan.tasks)

    def groups(self, plan: CollaborationPlan) -> List[List[AgentTask]]:
        """返回可并行的层级分组。"""
        return topological_levels(plan.tasks)
