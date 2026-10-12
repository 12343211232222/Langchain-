"""GRAPH_ENGINE 灰度开关（LangGraph 迁移 P0：基建 + 冒烟）。

- ``get_graph_engine()``：读取 ``Settings.basic_settings.GRAPH_ENGINE``
  （basic_settings.yaml 可配置），取值 legacy / langgraph，默认 legacy。
- ``is_langgraph_enabled()``：是否启用 LangGraph 编排路径。
- ``graph_engine_smoke()``：M06 Runtime 冒烟测试（Input→Output 最小图），
  用于验证 langgraph 引擎在当前进程可用。

注意：本模块只提供开关与冒烟，不改变现有 AgentExecutor 生产链路；
后续迁移逐步在各入口按 ``is_langgraph_enabled()`` 分流。
"""

from chatchat.core.logging import get_logger

logger = get_logger(__name__)

LEGACY = "legacy"
LANGGRAPH = "langgraph"


def get_graph_engine() -> str:
    """当前编排引擎标识。"""
    from chatchat.settings import Settings

    engine = (Settings.basic_settings.GRAPH_ENGINE or LEGACY).strip().lower()
    if engine not in (LEGACY, LANGGRAPH):
        logger.warning(f"未知 GRAPH_ENGINE={engine}，回退 legacy")
        return LEGACY
    return engine


def is_langgraph_enabled() -> bool:
    """是否启用 LangGraph 编排路径。"""
    return get_graph_engine() == LANGGRAPH


def graph_engine_smoke() -> bool:
    """M06 Runtime 冒烟：构建最小 Input→Output 图并执行一次。"""
    from chatchat.agent.runtime import (
        GraphBuilder,
        InputNode,
        LangGraphRuntime,
        OutputNode,
    )

    graph = (
        GraphBuilder()
        .add_node("input", InputNode())
        .add_node("output", OutputNode())
        .set_entry("input")
        .add_edge("input", "output")
        .build()
    )
    result = LangGraphRuntime(graph).run("smoke")
    ok = result.success and result.output == "smoke"
    logger.info(f"graph engine smoke: {'OK' if ok else 'FAILED'}")
    return ok
