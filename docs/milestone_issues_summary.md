# 里程碑问题清单（M01–M10）

> 生成时间：2026-10-08  
> 对照目标：**能力集成 + 接口解耦 + Agent 插件化**  
> 目标架构：
>
> - `LangChain-CC → 统一 RAG Service → RAGFlow Parser / Retrieval 能力`
> - `LangChain-CC → LangGraph Runtime → Agent Registry → { TenderReviewAgent, PRDTestAgent, FutureAgent }`
>
> 证据来源：代码静态检索、模块导入验证、全量单测执行（203 passed / 0 failed）、`chatchat/server/` 接线检索

---

## 一、目标 vs 现状总览

| 目标            | 现状                                                                | 差距                                                               |
| ------------- | ----------------------------------------------------------------- | ---------------------------------------------------------------- |
| **能力集成**      | RAG 检索/解析/切分已接进主链；`ToolManager` 已桥接老 `tools_registry`             | RAGFlow 集成度 **0%**；Agent 侧接入度 **0%**；主服务仍走老 `AgentExecutor`      |
| **接口解耦**      | M02 有 `get_rag_service()` 工厂，消费方只认接口                              | M03/M04 **无工厂**，消费方直接 import 具体类；M05 默认关闭                        |
| **Agent 插件化** | M08 Registry + Factory 机制已实现，`register_tender_review_agent()` 可注册 | 注册表仅内存态、无动态发现、无对外入口；仅 1 个业务 Agent；**PRDTestAgent 不存在**；M10 并行是空壳 |

---

## 二、逐里程碑问题清单

### M01 Core 基础层

| #   | 问题              | 证据                                                                                          | 影响                      |
| --- | --------------- | ------------------------------------------------------------------------------------------- | ----------------------- |
| 1.1 | **未被主服务采纳**     | `chatchat/server/` 中 `chatchat.core` 零命中；主服务仍用自带的 `chatchat/server/utils.py:build_logger()` | 日志/异常两套体系并存，「统一基础层」名不副实 |
| 1.2 | 仅在 M02–M10 内部使用 | 只有新模块 `from chatchat.core.logging import get_logger`                                        | 基础设施孤岛化，老代码改造收益为零       |

### M02 统一 RAG Service

| #   | 问题                | 证据                                                                         | 影响                                     |
| --- | ----------------- | -------------------------------------------------------------------------- | -------------------------------------- |
| 2.1 | **只接了「知识库对话」一条路** | 仅 `server/chat/kb_chat.py:20,80-88` 引用；`server/chat/file_chat.py`（文件对话）未接入 | 解耦只做了一半，文件对话仍走老逻辑                      |
| 2.2 | RAGFlow 毫无踪迹      | 全仓库 `ragflow` 仅 6 处**注释**，无依赖/客户端/配置                                       | 目标链路 `统一 RAG Service → RAGFlow` **0%** |
| 2.3 | 抽象只覆盖 retrieval   | `RAGServiceInterface` 只有 `retrieve` 一个方法                                   | 生成、引用（Citation）等能力未纳入统一层               |

### M03 Document Parser

| #   | 问题                       | 证据                                                                                                      | 影响                                     |
| --- | ------------------------ | ------------------------------------------------------------------------------------------------------- | -------------------------------------- |
| 3.1 | **没有工厂/provider，接口形同虚设** | `server/knowledge_base/utils.py:334` 直接 `from chatchat.rag.parser.service import CurrentDocumentParser` | 换 RAGFlow DeepDoc 必须改**业务代码**，违背「接口解耦」 |
| 3.2 | 无 provider.py            | `chatchat/rag/` 下只有 `provider.py` 一个工厂，且只管 RAG Service                                                  | 三个插槽只做了一个                              |

### M04 Chunk / Ingestion

| #   | 问题             | 证据                                                                                              | 影响                       |
| --- | -------------- | ----------------------------------------------------------------------------------------------- | ------------------------ |
| 4.1 | **同 M03，无工厂**  | `server/knowledge_base/utils.py:353` 直接 `from chatchat.rag.chunk.service import CurrentChunker` | 换 RAGFlow Chunker 需改业务代码 |
| 4.2 | 与 M03 耦合在同一个类里 | `FileType` 风格类同时调 parser 和 chunker                                                              | 想单独替换其中一个，拆不开            |

### M05 Hybrid Retrieval

| #   | 问题                    | 证据                                                                                                                | 影响                               |
| --- | --------------------- | ----------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| 5.1 | **默认关闭，线上永远走 vector** | `rag/service.py:28` `strategy = request.retrieval_strategy or VECTOR`；而 `kb_chat.py` 调用 `RAGQuery(...)` **未传该字段** | BM25 + RRF/Weighted 融合全部白做，线上零收益 |
| 5.2 | **Rerank 是空实现**       | `rag/retrieval/rerank.py` 只有 `NoOpReranker`，模块自述「不加载 BGE / Cross Encoder / RAGFlow Reranker」                      | 只有多路召回，没有重排，召回质量提升有限             |
| 5.3 | 融合策略无真实调优             | `RRFFusion` / `WeightedFusion` 权重为硬编码默认值                                                                          | 效果未验证                            |

### M06 LangGraph Runtime

| #   | 问题                                    | 证据                                                                                                                      | 影响                                                             |
| --- | ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| 6.1 | **全同步，零异步**                           | `chatchat/agent/` 下 `async def` / `astream` / `ainvoke` / `asyncio` **全部 0 命中**；`LangGraphRuntime.run` 用 `graph.invoke` | 接入 FastAPI 会**阻塞事件循环**；且无流式，与主服务 SSE 架构根本冲突 —— **这是接入的最大技术障碍** |
| 6.2 | **GraphBuilder 硬编码 `AgentState`，不泛型** | `runtime/graph.py:28` `StateGraph(AgentState)`；`tender_review/graph.py:34` 被迫直接用 `StateGraph(TenderReviewState)`        | M06 想统一的构图入口没统一，出现两套构图风格                                       |
| 6.3 | 无 Streaming、无 Checkpoint              | 模块自述「明确排除 Streaming 重写 / Checkpoint 实现」                                                                                 | 多轮记忆无持久化，无法中断恢复                                                |
| 6.4 | `tool_results` 只是预留字段                 | `state.py:30` 注释「M06 仅预留字段」                                                                                             | **没有 agent↔tools 的 ReAct 循环**，图能「跑通」但不具备真正的工具调用 Agent 能力       |
| 6.5 | 未接线                                   | `server/` 无 `from chatchat.agent`                                                                                       | 154 个测试覆盖的能力无法被用户触发                                            |

### M07 Agent Harness

| #   | 问题              | 证据                                                                      | 影响                               |
| --- | --------------- | ----------------------------------------------------------------------- | -------------------------------- |
| 7.1 | Memory 只有内存实现   | `harness/memory.py` 仅 `InMemoryMemoryManager`，自述「单次执行 / session 级短期上下文」 | 无法跨会话、跨进程                        |
| 7.2 | Policy 形同虚设     | `harness/policy.py` 仅 `AllowAllPolicy`，自述「不开发完整企业安全系统」                  | 无任何实际的执行约束/审批能力                  |
| 7.3 | 同步，无 async      | 同 6.1                                                                   | 同上                               |
| 7.4 | **亮点**：真正桥接了老代码 | `harness/tools.py` 延迟 import 既有 `tools_registry` 与 `_TOOLS_REGISTRY`    | 这是 M01–M10 中唯一一条「新框架 → 老代码」的实质通道 |
| 7.5 | 未接线             | 同 6.5                                                                   | —                                |

### M08 Agent Registry

| #   | 问题             | 证据                                      | 影响                                                    |
| --- | -------------- | --------------------------------------- | ----------------------------------------------------- |
| 8.1 | **只有进程内内存注册表** | `registry.py` 仅 `InMemoryAgentRegistry` | 服务重启即丢失注册，插件化没有持久层                                    |
| 8.2 | **无动态插件发现**    | 模块自述「明确不做：动态插件系统、分布式 Registry」          | 目前靠代码显式调用 `register_tender_review_agent()`，不是配置/发现式插件 |
| 8.3 | 无对外暴露          | 无 HTTP 端点、无 CLI 列出/调用 Agent             | 「插件化」目前只是类库能力，不是产品能力                                  |

### M09 TenderReviewAgent

| #   | 问题                   | 证据                                                                             | 影响                   |
| --- | -------------------- | ------------------------------------------------------------------------------ | -------------------- |
| 9.1 | **是线性 DAG，不是 Agent** | `tender_review/graph.py` 7 节点**全 add_edge 顺序连接**，无分支、无循环、无重试回边                 | 没有决策与自我纠错，本质是一段固定流水线 |
| 9.2 | 无运行入口                | 无 `__main__` / CLI / API 端点                                                    | 用户无法触发               |
| 9.3 | 依赖手工组装               | `agent.py:75` `rag_service` 为 None 直接 `raise TenderReviewError`                | 必须外部显式注入依赖，无自动装配     |
| 9.4 | 测试用全 Mock            | `test_tender_review_agent.py` 首行「Mock RAG / Mock LLM / Mock Harness，不调用真实 LLM」 | 从未与真实 LLM + 真实知识库联调过 |

### M10 Multi-Agent Collaboration

| #    | 问题                        | 证据                                                                                                                   | 影响                 |
| ---- | ------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------ |
| 10.1 | **ParallelStrategy 是假的**  | `strategies.py:87-88` `order()` 直接 `return topological_order(plan.tasks)`，与 Sequential **完全相同**；只多一个未被使用的 `groups()` | 名为并行，实为串行，没有任何并行执行 |
| 10.2 | Coordinator 默认串行          | `coordinator.py:46` `self.strategy = strategy or SequentialStrategy()`                                               | 并行策略实际上没被跑起来       |
| 10.3 | **Demo Agent 不是生产 Agent** | `demo_agents.py:1` 自述「仅用于 Demo 与测试，**非完整生产 Agent**」；无 llm 时走确定性规则                                                    | 不能作为上线 Agent 使用    |
| 10.4 | 未接线                       | 同 6.5                                                                                                                | —                  |

---

## 三、跨模块 / 工程性问题

| # | 问题                   | 证据                                                                                                                                                                 | 严重度                                                                                 |
| - | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| A | **迁移方案一步没走**         | `GRAPH_ENGINE` 灰度开关全仓库 **0 命中**（含 yaml）；无 `create_react_agent` / `astream_events` / `MemorySaver` / `SqliteSaver`                                                  | **P0** — `docs/langgraph_migration_plan.md` 的 P0 都没启动，老 `AgentExecutor` while 循环仍在线 |
| B | **Agent 侧无任何入口**     | `chatchat/agent/` 无 `__main__`/argparse/click；无脚本引用；`server/` 无 import                                                                                             | **P0** — M06–M10（154 测试）全部无法被用户触发                                                   |
| C | **PRDTestAgent 不存在** | 全仓库 `prd.?test` 零命中                                                                                                                                                | **P0（相对你的目标）** — 目标架构里的第二个业务 Agent 还没影子                                             |
| D | 测试基建损坏               | `pyproject.toml:234` addopts 含 `--snapshot-warn-unused`（syrupy 未装）→ 直接 `pytest` 报错；`asyncio_mode` 未注册 + pytest-asyncio 缺失 → 原仓库 `test_mcp_prompts.py` 1 例 async 失败 | **P1**                                                                              |
| E | 双轨并存成本高              | 两套日志（core vs server.utils）、两套异常、两套构图（GraphBuilder vs 裸 StateGraph）、两套 Agent（新框架 vs PlatformToolsAgentExecutor）、两套检索入口                                              | **P2**                                                                              |

---

## 四、按目标归因

### 目标一：能力集成

- ✅ 已完成：M02/M03/M04 接入主链；M07 `ToolManager` 桥接老 `tools_registry`
- ❌ 未完成：**RAGFlow 0%**、**Agent 侧 0%**、主服务仍走老 AgentExecutor

### 目标二：接口解耦

- ✅ 已完成：M02 的 `RAGServiceInterface` + `get_rag_service()` 工厂，消费方只认接口
- ❌ 未完成：**M03/M04 抽了接口但没做工厂**，消费方直接 import 具体类 → 换实现要改业务代码
- ❌ 未完成：M05 默认关闭 → 新能力解耦了但用不上

### 目标三：Agent 插件化

- ✅ 已完成：M08 的 `AgentRegistryInterface` + `InMemoryAgentRegistry` + `CallableAgentFactory` + `HarnessAgent` 适配器，机制齐备
- ❌ 未完成：
  - 无持久化注册表（重启丢失）
  - 无动态发现/配置式加载（自述不做插件系统）
  - 无对外入口（HTTP/CLI）
  - 只有 1 个业务 Agent，PRDTestAgent 缺失
  - M10 并行能力是空壳，多 Agent 协作实际仍是串行

---

## 五、建议修复优先级

| 优先级    | 事项                                          | 理由                          |
| ------ | ------------------------------------------- | --------------------------- |
| **P0** | 给 M06 补 `async` 执行路径（`ainvoke` / `astream`） | 不解决就无法接入 FastAPI/SSE，其余都是空谈 |
| **P0** | 给 Agent 侧加最小可运行入口（CLI 或 API 端点）             | 让 M06–M10 真正「通电」，否则永远只是类库   |
| **P0** | 补 M03/M04 的 provider 工厂                     | 直接服务「接口解耦」目标，工作量小、收益立竿见影    |
| **P1** | 实现 `GRAPH_ENGINE` 灰度开关，启动 M06 迁移 P0         | 让新老编排可切换、可回滚                |
| **P1** | 在 `kb_chat.py` 暴露 `retrieval_strategy` 参数   | 让 M05 混合检索真正可用              |
| **P1** | GraphBuilder 泛型化（`GraphBuilder[StateT]`）    | 消除两套构图风格                    |
| **P1** | 实现真实 Reranker（BGE / Cross Encoder）          | 混合检索才有质量收益                  |
| **P2** | Redis/SQLite 持久化 Registry + 配置式 Agent 加载    | 支撑真正的「插件化」                  |
| **P2** | 实现 M10 真实并行（`concurrent.futures`）           | 当前 `ParallelStrategy` 名不副实  |
| **P2** | 修复 pytest addopts / asyncio 配置              | 恢复「一条命令跑全量测试」               |
| **P2** | 新增 **PRDTestAgent**                         | 补齐你目标架构中的第二个业务 Agent        |

---

## 六、一句话总结

**基础设施（M01–M10）模块级完成度很高：10 个模块全部可导入、203 个单测全绿。但三个目标都只完成了一半——**

- **能力集成**：RAG 侧接进去了，RAGFlow 和 Agent 侧都没接；
- **接口解耦**：只有 RAG Service 一个插槽做对了，Parser/Chunker 是有接口没工厂；
- **Agent 插件化**：注册机制建好了，但没有入口、没有持久化、没有第二个 Agent，多 Agent 并行还是假的。

**根因是同一个：M06–M10 这一整套 Agent 框架从头到尾没有任何一个「对外出口」，导致它只能被单测验证，无法被使用。**
