# LangGraph 迁移方案

> 目标：把 Langchain-Chatchat 的「编排层」从 LangChain 的 `Chain` + `AgentExecutor` 手写循环，迁移到 LangGraph 的显式 `StateGraph`。
> 约束：对外 API / WebUI / SSE 事件协议保持不变；底层 LangChain 组件（LLM、Embedding、VectorStore、Retriever、Tool）全部复用，只替换编排。

---

## 1. 迁移目标与原则

- **换编排，不换组件**：`get_ChatOpenAI` / `ChatPlatformAI`、embedding、FAISS/Milvus、`file_rag/retrievers`、`tools_factory`、prompt 配置等均复用。
- **对前端透明**：Streamlit WebUI、OpenAI 兼容 API、python-sdk 的入参/出参不变，SSE 事件结构（`PlatformToolsAction` / `PlatformToolsFinish` / `PlatformToolsLLMStatus` 等）保持原样。
- **增量 + 可回滚**：分阶段落地，每阶段有独立验收，失败可切回旧实现。

## 2. 现状编排层梳理

项目当前有两条独立的编排路径：

### 2.1 RAG 链路（非 Agent）
- 入口：`libs/chatchat-server/chatchat/server/chat/kb_chat.py`（知识库对话）、`file_chat.py`（文件对话）
- 链路：`search_docs` 检索 → 拼 `context` → `ChatPromptTemplate | llm` 线性链 → 生成
- 关键点：检索在函数内手动调用，生成是单条 `chain`；流式靠 `AsyncIteratorCallbackHandler`

### 2.2 Agent 执行循环
- 入口：`libs/chatchat-server/chatchat/server/chat/chat.py`
- 组装：`PlatformToolsRunnable.create_agent_executor()`（`langchain_chatchat/agents/platform_tools/base.py`）
- 执行：`PlatformToolsAgentExecutor._call/_acall` 里的 `while self._should_continue(...)` 循环（`langchain_chatchat/agents/all_tools_agent.py`）
- 流式：`AgentExecutorAsyncIteratorCallbackHandler` 收集 token/工具事件，在 `PlatformToolsRunnable.invoke()` 里映射成 schema 对象
- 记忆：`intermediate_steps` 序列化后存 DB（`message.metadata`），下一轮由 `create_models_chains` 读回

## 3. 迁移总览（三层）

| 层 | 现状 | LangGraph 目标 | 改动性质 |
|----|------|----------------|----------|
| 底层组件 | LLM/embedding/vectorstore/retriever/tool | 复用 | 不动 |
| RAG 编排 | `chain = prompt \| llm` | `StateGraph`：`retrieve → rerank? → generate` | 重写链路 |
| Agent 编排 | `AgentExecutor` while 循环 | `create_react_agent` / 手写 StateGraph | 重写循环 |
| 横切 | callback 流式、DB 中间步骤 | `astream_events`、checkpointer | 适配 |

## 4. 详细设计

### 4.1 依赖变更
- 在 `libs/chatchat-server/pyproject.toml` 增加依赖：
  - `langgraph`（StateGraph、create_react_agent、checkpoint）
  - 补充 `langgraph-checkpoint-sqlite`（若用 SqliteSaver 持久化记忆）
- Poetry 锁定后 `poetry lock` 重新生成。

### 4.2 RAG 链路图化

**新建** `langchain_chatchat/graphs/rag_graph.py`：

```python
from typing import TypedDict
from langgraph.graph import StateGraph, END

class RAGState(TypedDict):
    query: str
    docs: list          # 检索结果
    context: str        # 拼好的上下文
    answer: str         # 最终回答

# 节点
def retrieve(state): ...       # 复用 search_docs / search_temp_docs
def rerank(state): ...         # 可选，接入现有 reranker
def generate(state): ...       # 复用 get_ChatOpenAI + prompt

# 构图
builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve)
builder.add_node("rerank", rerank)      # 可选，据 USE_RERANKER 开关决定是否加边
builder.add_node("generate", generate)
builder.set_entry_point("retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
```

**改动** `kb_chat.py` / `file_chat.py`：
- 删除 `chain = chat_prompt | llm`，改为 `graph.astream({"query": query, ...})`。
- 流式改用 `graph.astream(..., stream_mode="updates")` 或对 `generate` 节点用 `astream_events` 取 token。
- `format_reference`、`search_docs` 等逻辑原样保留。

### 4.3 Agent 执行循环图化（核心）

**两种实现，二选一**：

1. **快速方案（推荐先做）**：用 LangGraph 预置的 `create_react_agent(llm, tools, checkpointer=...)`，直接产出 `react` 图。
2. **精细方案**：手写 `StateGraph`，节点 = `agent`（llm + bind_tools）+ `tools`（执行工具），`conditional_edge` 判断 `AgentFinish`。

**新建** `langchain_chatchat/graphs/agent_graph.py`：

```python
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

def build_agent_graph(llm, tools, checkpointer=None):
    return create_react_agent(
        llm, tools,
        checkpointer=checkpointer or MemorySaver(),
    )
```

**替代对象**：
- `PlatformToolsAgentExecutor._call/_acall` 的 while 循环 → 图执行
- `agents_registry.py` 各分支 → 由「图构建器 + prompt 模板」替代，`agent_type` 退化为「prompt 选择 + 是否用 platform 工具协议」

**工具准备**：`create_agent_executor` 里的三类工具——`temp_tools`（自定义 BaseTool）、`assistants_builtin_tools`（AdapterAllTool）、`mcp_tools`——统一用 `llm.bind_tools(tools)` 转为 function schema 后交给 `create_react_agent`。`AdapterAllTool`（web_browser / drawing 等）需要新增 LangGraph 工具适配或从 graph 中特殊化处理。

### 4.4 流式事件适配（改动量大，最易出错）

现状是 `callback.aiter()` 读队列 → 映射 `AgentStatus` → schema。LangGraph 用 `astream_events`：

| AgentStatus | LangGraph 事件 | 映射目标 |
|-------------|----------------|----------|
| `llm_start` / `llm_end` | `on_chat_model_start` / `on_chat_model_end` | `PlatformToolsLLMStatus` |
| `llm_new_token` | `on_chat_model_stream` | `PlatformToolsLLMStatus` |
| `agent_action` / `tool_start` | `on_tool_start` | `PlatformToolsAction` / `PlatformToolsActionToolStart` |
| `tool_end` | `on_tool_end` | `PlatformToolsActionToolEnd` |
| `agent_finish` | `on_chain_end`（graph 层） | `PlatformToolsFinish` |

**实现**：新建适配层，把 `graph.astream_events(config, version="v2")` 的事件流，翻译回现有 schema 对象，供 `chat.py` 的 `chat_iterator_event` 消费。**前端与 `OpenAIChatOutput` 结构零改动**。

### 4.5 记忆 / 中间步骤适配

- 现状：`intermediate_steps` 存 DB `message.metadata`。
- 方案：改用 `SqliteSaver`（`langgraph-checkpoint-sqlite`）持久化图 State；`conversation_id` 作为 `thread_id`。
- 收益：LangGraph 自带的检查点/恢复能力，替代手动的 `loads/dumps`。
- 兼容：`create_models_chains` 里读 `intermediate_steps` 的逻辑改为读 checkpoint 或删除。

### 4.6 目录规划（迁移后）

```
langchain_chatchat/
  graphs/                    # 新：LangGraph 图定义
    rag_graph.py             # RAG 链
    agent_graph.py           # Agent 循环
    graph_events.py          # astream_events -> 现有 schema 的适配器
  agents/                    # 旧：保留 prompt 模板与 output_parser（create_react_agent 需 prompt 复用）
  agent_toolkits/            # 旧：工具定义复用
chatchat/server/
  chat/chat.py               # 改：消费新事件流
  chat/kb_chat.py            # 改：走 rag_graph
  chat/file_chat.py          # 改：走 rag_graph
```

## 5. 分步实施计划

> 灰度开关：迁移全程以 `GRAPH_ENGINE`（`Settings` 新增布尔项，默认 `legacy`/`false`）控制新旧实现切换，`kb_chat` / `chat` 入口按开关分发到旧链或新图。每个阶段完成后打开开关做小范围验证，异常即切回，保证任意时刻可回滚。

| 阶段 | 内容 | 交付物 | 验收标准 |
|------|------|--------|----------|
| P0 基建+冒烟 | 加 `langgraph` 依赖；新增 `GRAPH_ENGINE` 开关；写最小 StateGraph 冒烟（多节点 + `astream`） | `graphs/` 骨架、开关、冒烟用例 | 服务启动正常；冒烟图可 `astream`；旧链路零影响 |
| P1 RAG 链路 | P1a 先非流式跑通 `retrieve→generate`；P1b 再接 `astream` 流式 | `rag_graph.py`；改造 `kb_chat.py` / `file_chat.py` | 知识库问答结果与旧版一致；流式 token 顺序一致 |
| P2 Agent 循环+事件 | P2a `create_react_agent` 跑通普通 `BaseTool`；P2b `graph_events.py` 把 `astream_events` 映射回 `PlatformToolsAction/Finish/LLMStatus` | `agent_graph.py`、`graph_events.py`；改造 `chat.py` | 单/多工具结果一致；WebUI 工具过程可视化一致 |
| P3 内置平台工具+MCP | 处理 `AdapterAllTool`（web_browser/drawing 等返回 `AgentFinish` 的特化逻辑）与 MCP 工具接入 | 平台工具适配层 | 文生图/浏览器/MCP 工具正常 |
| P4 记忆持久化 | `SqliteSaver` 替换 DB `intermediate_steps`，`thread_id=conversation_id` | checkpointer 接入 | 多轮记忆、中断恢复正常 |
| P5 清理回归 | 移除旧 `AgentExecutor` / callback handler / 旧 `chain` 路径；跑全量测试 | 删除旧编排代码 | 旧路径无引用；`tests/` 全绿 |

## 6. 风险与回滚

- **流式事件语义差异**：LangGraph 的事件颗粒度与 `AgentStatus` 不完全一一对应，P2b 是最大风险点，需逐事件核对后才切换开关。
- **AdapterAllTool（内置平台工具）**：web_browser / drawing 等返回 `AgentFinish` 的特化逻辑在 LangGraph 下需重写，P3 专项处理；P2 期间先只支持普通 `BaseTool`。
- **版本兼容**：LangGraph 与现有 LangChain / langchain-openai 版本可能冲突，P0 冒烟必须覆盖 `astream_events` 与 `@tool` 两条路径。
- **回滚**：全程依赖 `GRAPH_ENGINE` 开关，任意阶段出问题切回旧路径（`chat.py` 旧 executor / `kb_chat.py` 旧 chain）即可，无需回滚代码。

## 7. 关键文件清单

- `libs/chatchat-server/chatchat/server/chat/chat.py` — Agent 对话入口
- `libs/chatchat-server/chatchat/server/chat/kb_chat.py` — RAG 对话入口
- `libs/chatchat-server/chatchat/server/chat/file_chat.py` — 文件对话入口
- `libs/chatchat-server/langchain_chatchat/agents/all_tools_agent.py` — AgentExecutor 循环（待替换）
- `libs/chatchat-server/langchain_chatchat/agents/platform_tools/base.py` — Runnable 组装与事件映射（待替换）
- `libs/chatchat-server/langchain_chatchat/agents/platform_tools/schema.py` — SSE 事件协议（保持不变）
- `libs/chatchat-server/chatchat/server/agents_registry/agents_registry.py` — agent 类型注册（待重构）
- `libs/chatchat-server/chatchat/server/file_rag/retrievers/` — 检索器（复用）
- `libs/chatchat-server/pyproject.toml` — 依赖声明（新增 langgraph）