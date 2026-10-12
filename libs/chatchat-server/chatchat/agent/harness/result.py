"""Harness 统一结果对象。

``HarnessResult`` 位于 ``RuntimeResult``（M06）上层：
LangGraphRuntime -> RuntimeResult -> AgentHarness -> HarnessResult。

不修改 M06 ``RuntimeResult``。
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from chatchat.agent.runtime.result import RuntimeResult


@dataclass
class HarnessResult:
    """一次 ``AgentHarnessInterface.execute`` 的统一返回。

    参数:
        success: 是否成功。
        output: 最终输出文本。
        context: 执行结束后的 ``HarnessContext``。
        runtime_result: 最后一次 M06 Runtime 的原始结果。
        retry_count: 实际重试次数。
        error: 失败原因（成功时为 None）。
        metadata: 执行元数据（``ExecutionMetadata`` 的 dict 形式）。
    """

    success: bool
    output: Optional[str] = None
    context: Optional[Any] = None
    runtime_result: Optional[RuntimeResult] = None
    retry_count: int = 0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
