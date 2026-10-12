"""Harness 层异常。

复用 M01 异常体系（``ChatchatError``）。
"""

from chatchat.core.exceptions import ChatchatError


class HarnessError(ChatchatError):
    """Agent Harness 执行期间的统一异常。"""


class PolicyViolationError(HarnessError):
    """Policy 检查未通过（输入非法、执行被拦截等）。"""
