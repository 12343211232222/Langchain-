"""Memory Manager：Memory 抽象与内存实现。

M07 只建立接口与最小实现（单次执行 / session 级短期上下文），
不引入 Redis / 数据库 / 向量库 / 长期 Memory Service。

Memory 是 Harness 的扩展点，不是 M07 的核心业务能力。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class MemoryManager(ABC):
    """Memory 生命周期接口（按 session_id 组织）。"""

    @abstractmethod
    def create(self, session_id: str) -> None:
        """为 session 创建空记忆。"""
        raise NotImplementedError

    @abstractmethod
    def update(self, session_id: str, record: Dict[str, Any]) -> None:
        """向 session 追加一条记忆记录。"""
        raise NotImplementedError

    @abstractmethod
    def get(self, session_id: str) -> List[Dict[str, Any]]:
        """读取 session 全部记忆记录。"""
        raise NotImplementedError

    @abstractmethod
    def load(self, session_id: str) -> List[Dict[str, Any]]:
        """加载 session 记忆（内存实现等同 get）。"""
        raise NotImplementedError

    @abstractmethod
    def save(self, session_id: str, record: Dict[str, Any]) -> None:
        """保存一条记忆（内存实现等同 update）。"""
        raise NotImplementedError

    @abstractmethod
    def clear(self, session_id: str) -> None:
        """清空 session 记忆。"""
        raise NotImplementedError


class InMemoryMemoryManager(MemoryManager):
    """进程内 Memory：dict[session_id, list[record]]。"""

    def __init__(self):
        self._store: Dict[str, List[Dict[str, Any]]] = {}

    def create(self, session_id: str) -> None:
        self._store.setdefault(session_id, [])

    def update(self, session_id: str, record: Dict[str, Any]) -> None:
        self.create(session_id)
        self._store[session_id].append(record)

    def get(self, session_id: str) -> List[Dict[str, Any]]:
        return list(self._store.get(session_id, []))

    def load(self, session_id: str) -> List[Dict[str, Any]]:
        return self.get(session_id)

    def save(self, session_id: str, record: Dict[str, Any]) -> None:
        self.update(session_id, record)

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)
