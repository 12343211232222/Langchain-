"""项目级基础异常体系。

M01 阶段仅建立轻量异常基类与若干基础异常，
供后续 RAG Service、LangGraph Runtime、Agent Harness 等模块复用。

设计原则：
- 不吞掉原始异常（不要 `except Exception: raise ChatchatError("...")`）。
- 需要包装时使用 `raise ServiceError("...") from exc` 保留异常链。
- 本阶段只建立基础设施，不强制迁移现有代码。
"""


class ChatchatError(Exception):
    """项目基础异常。"""


class ConfigurationError(ChatchatError):
    """配置错误（配置文件缺失、字段非法、加载失败等）。"""


class ServiceError(ChatchatError):
    """服务执行错误。"""


class ResourceNotFoundError(ChatchatError):
    """资源不存在（知识库、文档、工具、模型等）。"""


class ValidationError(ChatchatError):
    """参数或输入校验错误。"""