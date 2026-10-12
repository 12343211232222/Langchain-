"""Runtime 层异常。

复用 M01 异常体系（``ChatchatError``）。命名避免与 Python 内置
``RuntimeError`` 冲突，故使用 ``GraphRuntimeError``。
"""

from chatchat.core.exceptions import ChatchatError


class GraphRuntimeError(ChatchatError):
    """Graph / Runtime 执行期间的统一异常。

    节点执行失败、图构建非法、超出步数上限等场景抛出。
    """
