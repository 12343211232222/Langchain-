"""Document Parser 统一获取入口（M03）。

消费方只认 ``DocumentParser`` 接口；按配置选择实现：
- ``RAGFLOW_PARSER_ENABLED=True``  → ``RAGFlowDocumentParser``（DeepDoc）
- 其他（默认 False / 未知值 / 配置缺失） → ``CurrentDocumentParser``（本地 loaders）
"""

from chatchat.core.logging import get_logger
from chatchat.rag.parser.interfaces import DocumentParser

logger = get_logger(__name__)

_parser: DocumentParser | None = None


def get_document_parser() -> DocumentParser:
    """获取当前 Document Parser（模块级单例）。"""
    global _parser
    if _parser is None:
        enabled = _is_ragflow_enabled()
        if enabled:
            try:
                from chatchat.rag.parser.service_ragflow import RAGFlowDocumentParser

                _parser = RAGFlowDocumentParser()
                logger.info("document parser: RAGFlowDocumentParser (DeepDoc)")
            except Exception as e:
                # 已开启但 RAGFlow 未配置完整时降级，不让摄取链路整体失败
                logger.warning(f"RAGFlow Parser 初始化失败（{e}），回退 CurrentDocumentParser")
                from chatchat.rag.parser.service import CurrentDocumentParser

                _parser = CurrentDocumentParser()
        else:
            from chatchat.rag.parser.service import CurrentDocumentParser

            _parser = CurrentDocumentParser()
    return _parser


def _is_ragflow_enabled() -> bool:
    """读取开关；未知值/异常回退 False 并打 warning。"""
    from chatchat.settings import Settings

    value = getattr(Settings.basic_settings, "RAGFLOW_PARSER_ENABLED", False)
    if isinstance(value, bool):
        return value
    logger.warning(f"RAGFLOW_PARSER_ENABLED 非法值 {value!r}，回退 False")
    return False
