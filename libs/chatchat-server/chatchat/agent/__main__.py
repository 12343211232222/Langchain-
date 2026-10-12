"""ChatChat Agent 框架最小可运行入口（CLI）。

用法（在 libs/chatchat-server 目录下）::

    # 列出已注册 Agent
    python -m chatchat.agent list

    # 运行标书审查 Agent（需要知识库中有数据时传 --kb）
    python -m chatchat.agent run tender_review \
        --requirement "项目经理必须具有5年以上相关项目经验。" \
        --response "项目经理张三具有7年相关项目经验。" \
        --kb my_kb

    # 运行 M10 多 Agent 协作 Demo
    python -m chatchat.agent collab \
        --input "项目经理必须具有5年以上相关项目经验。" --kb my_kb

说明：
- 入口通过 M08 Registry 发现与创建 Agent，经 M07 Harness → M06 Runtime 执行。
- 不传 --kb 时检索节点会跳过知识库检索（符合性判断将降级为 UNKNOWN）。
"""

import argparse
import json
import sys


def _build_registry():
    """构建 Registry 并注册已知 Agent。"""
    from chatchat.agent.registry import InMemoryAgentRegistry
    from chatchat.agent.tender_review import register_tender_review_agent
    from chatchat.agent.collaboration import register_demo_agents
    from chatchat.rag.provider import get_rag_service

    rag_service = get_rag_service()
    registry = InMemoryAgentRegistry()
    register_tender_review_agent(registry, rag_service=rag_service)
    register_demo_agents(registry, rag_service=rag_service)
    return registry


def cmd_list(args) -> int:
    registry = _build_registry()
    for d in registry.list():
        print(f"{d.agent_id}\t{d.name}\tv{d.version}\t{d.description}")
    return 0


def cmd_run(args) -> int:
    from chatchat.agent.tender_review import TenderReviewAgent

    registry = _build_registry()
    if args.agent_id != "tender_review":
        print(f"当前 CLI 仅支持运行 tender_review，得到: {args.agent_id}",
              file=sys.stderr)
        return 2
    agent: TenderReviewAgent = registry.create(
        "tender_review", knowledge_base=args.kb
    )
    from chatchat.agent.tender_review import TenderReviewRequest

    review = agent.review(TenderReviewRequest(
        tender_requirement=args.requirement,
        bidder_response=args.response,
        knowledge_base=args.kb,
    ))
    print(json.dumps(review.to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_collab(args) -> int:
    from chatchat.agent.collaboration import (
        AgentCoordinator,
        CollaborationContext,
        build_tender_review_plan,
    )

    registry = _build_registry()
    coordinator = AgentCoordinator(registry)
    context = CollaborationContext(user_input=args.input)
    result = coordinator.execute(
        build_tender_review_plan(args.input, knowledge_base=args.kb), context
    )
    if result.success and "t4_report" in result.results:
        print(result.results["t4_report"].output)
    else:
        print(json.dumps({
            "success": result.success,
            "failed_task": result.failed_task,
            "error": result.error,
        }, ensure_ascii=False, indent=2))
        return 1
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="chatchat.agent", description="ChatChat Agent 框架 CLI 入口"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="列出已注册 Agent").set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="运行指定 Agent")
    p_run.add_argument("agent_id")
    p_run.add_argument("--requirement", required=True, help="招标要求文本")
    p_run.add_argument("--response", default="", help="投标响应文本")
    p_run.add_argument("--kb", default="", help="知识库名称（可选）")
    p_run.set_defaults(func=cmd_run)

    p_collab = sub.add_parser("collab", help="运行 M10 多 Agent 协作 Demo")
    p_collab.add_argument("--input", required=True, help="招标要求文本")
    p_collab.add_argument("--kb", default="", help="知识库名称（可选）")
    p_collab.set_defaults(func=cmd_collab)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
