"""Runtime 统一返回结构。"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from chatchat.agent.runtime.state import AgentState


@dataclass
class RuntimeResult:
    """一次 ``AgentRuntimeInterface.run`` 的统一返回。

    参数:
        success: 是否成功完成。
        output: 最终输出（通常是 ``final_output``）。
        state: 图执行结束后的完整 State。
        error: 失败原因（成功时为 None）。
        metadata: 附加信息（耗时等）。
    """

    success: bool
    output: Optional[str] = None
    state: Optional[AgentState] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
