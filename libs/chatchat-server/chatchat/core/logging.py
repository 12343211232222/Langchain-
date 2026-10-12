"""统一日志入口（Facade）。

复用项目现有 loguru 配置（``chatchat.utils.build_logger``），
为未来 Agent、RAG Service、LangGraph Runtime 提供一致的 logger 获取接口。

本阶段不做任何日志系统重构，不修改日志格式、路径与级别。
"""

from chatchat.utils import build_logger


def get_logger(name: str | None = None):
    """获取统一 logger。

    复用项目现有 loguru 配置：
    - 不传 ``name``：返回默认全局 logger（不额外写文件）。
    - 传入 ``name``：按名称生成独立日志文件（与 ``build_logger`` 行为一致）。

    用法::

        from chatchat.core.logging import get_logger
        logger = get_logger(__name__)
        logger.info(...)
        logger.warning(...)
        logger.error(...)
        logger.exception(...)
    """
    return build_logger(name)