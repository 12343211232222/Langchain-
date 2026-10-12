# 里程碑完成度核对报告

> 核对时间：2026-10-08  
> 核对对象：`libs/chatchat-server`（Langchain-Chatchat 0.3.1.3 自研改造版）  
> 核对方法：代码静态检索 + 模块导入验证 + 全量单元测试执行  
> 环境：Python 3.10.11 / `.venv` / `PYTHONUTF8=1`

---

## 一、总体结论

项目共有 **10 个里程碑（M01–M10）**，全部有对应代码、全部有单测。

| 指标       | 结果                                    |
| -------- | ------------------------------------- |
| 里程碑数量    | 10（M01–M10，无 M11+）                    |
| 代码模块导入验证 | 10/10 全部通过，`__all__` 导出符号无缺失          |
| 里程碑单测    | **203 passed / 0 failed**（10 个测试文件）   |
| 已接入主服务   | **M02、M03、M04**（3 个）                  |
| 未接入主服务   | **M01（未采纳）、M05（可选未开启）、M06–M10（完全隔离）** |

**一句话总结**：M01–M10 作为「模块」全部按规格实现且测试全绿；但只有 RAG 侧（M02–M04）真正接进了线上链路，**Agent 侧 M06–M10 整套框架（约 191 个测试覆盖）完全没有接到 WebUI/API 上**，也没有任何运行入口。

---

## 二、逐项核对明细

| 里程碑     | 主题                        | 核心文件                                                          | 单测   | 接入主服务                                          | 完成度判定             |
| ------- | ------------------------- | ------------------------------------------------------------- | ---- | ---------------------------------------------- | ----------------- |
| **M01** | Core 基础层                  | `chatchat/core/`（exceptions / logging）                        | 5 ✅  | ⚠️ 仅被 M02–M10 内部使用，主服务未采纳                      | **完成**（模块级）       |
| **M02** | RAG 统一接口层                 | `chatchat/rag/`（schemas / interfaces / service / provider）    | 8 ✅  | ✅ `server/chat/kb_chat.py:20,80-88`            | **完成且已上线**        |
| **M03** | Document Parser           | `chatchat/rag/parser/`                                        | 7 ✅  | ✅ `server/knowledge_base/utils.py:334`         | **完成且已上线**        |
| **M04** | Chunk / Ingestion         | `chatchat/rag/chunk/`                                         | 9 ✅  | ✅ `server/knowledge_base/utils.py:353`         | **完成且已上线**        |
| **M05** | Hybrid Retrieval          | `chatchat/rag/retrieval/`（vector/bm25/fusion/rerank/strategy） | 20 ✅ | ⚠️ 已接入 `rag/service.py`，但主对话默认走 vector，混合检索未启用 | **完成**（能力就绪，默认关闭） |
| **M06** | LangGraph Runtime         | `chatchat/agent/runtime/`                                     | 22 ✅ | ❌ 未被 server 引用                                 | **完成**（独立模块）      |
| **M07** | Agent Harness             | `chatchat/agent/harness/`                                     | 35 ✅ | ❌ 未被 server 引用                                 | **完成**（独立模块）      |
| **M08** | Agent Registry            | `chatchat/agent/registry/`                                    | 22 ✅ | ❌ 未被 server 引用                                 | **完成**（独立模块）      |
| **M09** | Tender Review Agent       | `chatchat/agent/tender_review/`                               | 37 ✅ | ❌ 未接入，且无运行入口                                   | **完成**（独立模块）      |
| **M10** | Multi-Agent Collaboration | `chatchat/agent/collaboration/`                               | 38 ✅ | ❌ 未被 server 引用                                 | **完成**（独立模块）      |

### 测试执行原始结果

```
test_core                 5 passed
test_rag                  8 passed
test_parser               7 passed
test_chunk                9 passed
test_retrieval           20 passed
test_runtime             22 passed
test_harness             35 passed
test_agent_registry      22 passed
test_tender_review_agent 37 passed
test_multi_agent         38 passed
------------------------------------------------
里程碑小计               203 passed / 0 failed
```

---

## 三、各里程碑声明的交付范围（来自模块 docstring，已逐项验证存在）

- **M01**：`exceptions`（ChatchatError / ConfigurationError / ServiceError / ResourceNotFoundError / ValidationError）、`logging`（get_logger）
- **M02**：`schemas`(RAGQuery/RAGDocument/RAGResponse)、`interfaces`(RAGServiceInterface)、`service`(CurrentRAGService)、`provider`(get_rag_service)
- **M03**：`interfaces`(DocumentParser)、`service`(CurrentDocumentParser)
- **M04**：`interfaces`(Chunker)、`service`(CurrentChunker)
- **M05**：`schemas`/`interfaces`/`vector`(VectorRetriever)/`bm25`(BM25Retriever)/`fusion`(RRFFusion、WeightedFusion)/`rerank`(NoOpReranker)/`strategy`(RetrievalStrategy、HybridRetriever、build_hybrid_retriever)
- **M06**：`state`/`node`(Input/Passthrough/Output/RAG/LLM/Function)/`graph`(GraphBuilder、max_iterations_condition、build_rag_chat_graph)/`runtime`(AgentRuntimeInterface、LangGraphRuntime)/`result`/`exceptions`；**声明不做**：Harness/Registry/业务 Agent/复杂 Memory/Retry/Guardrail/Streaming 重写/Checkpoint 实现
- **M07**：`context`/`memory`(InMemory)/`tools`(适配 tools_registry)/`retry`/`policy`(AllowAll)/`execution`/`harness`(DefaultAgentHarness)/`result`/`exceptions`；**声明不做**：Registry/业务 Agent/Multi-Agent/长期 Memory
- **M08**：`schemas`/`interfaces`/`registry`(InMemoryAgentRegistry)/`factory`(CallableAgentFactory)/`exceptions`；**声明不做**：业务 Agent/Multi-Agent/动态插件/分布式 Registry
- **M09**：`agent`(TenderReviewAgent)/`graph`(7 节点线性图)/`nodes`/`state`/`schemas`/`prompts`；建立在 M01–M08 之上
- **M10**：`schemas`/`context`/`strategies`(Sequential/Parallel)/`coordinator`(AgentCoordinator)/`demo_agents`(Requirement/Retrieval/Compliance/Report)；**声明不做**：Supervisor/Planner/Debate/Reflection/分布式调度

---

## 四、发现的缺口与风险

### P0 — Agent 框架整体未接线

M06–M10 共 5 层、154 个单测覆盖的能力，**没有任何生产入口**：

- `chatchat/server/` 中精确匹配 `from chatchat.agent` / `import chatchat.agent` → **0 命中**（唯一疑似命中是 `langchain_chatchat.agents` 的子串误报）
- `chatchat/agent/` 下无 `__main__`、无 argparse/click 入口
- 无 `.py`/`.sh`/`.bat` 脚本引用该框架

结论：这部分能力目前只能通过单元测试验证，用户无法从 WebUI/API 触发。

### P0 — 迁移方案未启动

`docs/langgraph_migration_plan.md` 设计的灰度开关 `GRAPH_ENGINE` 在**全仓库 0 命中**（含 yaml）。方案的 P0（基建+冒烟）尚未落地，即"把老 `AgentExecutor` 循环迁到 LangGraph"这件事一步都没走。代码里同样搜不到 `create_react_agent`、`astream_events`、`MemorySaver`、`SqliteSaver`。

### P1 — M05 混合检索默认不生效

`chatchat/rag/service.py:28` 中 `strategy = request.retrieval_strategy or RetrievalStrategy.VECTOR.value`。而主入口 `kb_chat.py` 调用 `RAGQuery(...)` 时**未传 `retrieval_strategy`**，因此线上仍走纯向量检索。BM25 + RRF/Weighted 融合能力虽已实现并测试，但默认路径用不到。

### P1 — Rerank 是空实现

`chatchat/rag/retrieval/rerank.py` 仅有 `NoOpReranker`，模块声明"不加载 BGE / Cross Encoder / RAGFlow Reranker"。混合检索的实际召回质量提升因此受限。

### P2 — M01 未被主服务采纳

主服务仍使用自己的 `chatchat/server/utils.py:build_logger()`，未替换为 `chatchat.core.logging.get_logger`。M01 的异常体系与日志入口目前只在 M02–M10 内部使用，与既有代码存在两套日志/异常风格并存。

### P2 — 测试基建不可直接运行

- `pyproject.toml:234` 的 `addopts` 含 `--snapshot-warn-unused`（需 syrupy 插件），未安装 → 直接执行 `pytest` 会报 `unrecognized arguments`。本次核对通过 `-o addopts=""` 绕过。
- `asyncio_mode` 为未知配置项，pytest-asyncio 未安装 → `test_mcp_prompts.py::test_load_mcp_prompt` 失败（**该文件属原仓库自带，非里程碑交付**），`test_sdk_import` 正常。

---

## 五、修正后的完成度视图

如果把"完成度"定义为**面向用户可用**，而非"代码写完 + 单测过"：

| 层次      | 里程碑                         | 模块实现 | 线上可用        |
| ------- | --------------------------- | ---- | ----------- |
| RAG 侧   | M02 / M03 / M04             | 100% | ✅ 可用        |
| RAG 侧   | M05                         | 100% | ⚠️ 需显式传参才生效 |
| 基础      | M01                         | 100% | ⚠️ 未采纳      |
| Agent 侧 | M06 / M07 / M08 / M09 / M10 | 100% | ❌ 无入口、未接线   |
| 迁移      | 方案文档                        | 文档就绪 | ❌ P0 未启动    |

**建议的下一步**（按优先级）：

1. 给 M09/M10 加一个最小可运行入口（CLI 或 API 端点），让 Agent 框架真正"通电"；
2. 落实 `GRAPH_ENGINE` 灰度开关，为后续替换老 AgentExecutor 铺路；
3. 在 `kb_chat.py` 暴露 `retrieval_strategy` 参数，让 M05 混合检索可用；
4. 修 `pyproject.toml` 的 pytest addopts 与 asyncio 配置，恢复"一条命令跑全量测试"。
