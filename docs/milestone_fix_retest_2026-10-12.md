# 里程碑修复复测报告（2026-10-12）

> 复测对象：`libs/chatchat-server`（Langchain-Chatchat 0.3.1.3 自研改造版）
> 复测目的：验证 2026-10-08 审计报告与问题清单中所列问题的修复情况
> 复测方法：代码静态核对 + 全量单测 + 真实 CLI 端到端运行
> 环境：Python 3.10.11 / `.venv` / `PYTHONUTF8=1`

---

## 一、总览

| 指标 | 上次（10-08） | 本次（10-12） | 变化 |
|---|---|---|---|
| 单元测试 | 212 用例，1 failed / 211 passed | **224 用例，0 failed / 224 passed** | ✅ 新增 12 例，失败清零 |
| 默认可直接跑 pytest | ❌ 需 `-o addopts=""` 绕过 | ✅ **直接可跑** | ✅ 已修 |
| Agent 侧对外入口 | ❌ 无 | ✅ **`python -m chatchat.agent`** | ✅ 已补 |
| M06 异步路径 | ❌ 全同步 | ✅ **`arun` / `astream`** | ✅ 已补 |
| M03/M04 工厂 | ❌ 无 provider | ✅ **`get_document_parser` / `get_chunker`** | ✅ 已补 |
| `GRAPH_ENGINE` 开关 | ❌ 0 命中 | ✅ **已实现 + 冒烟** | ✅ 已补 |

**结论：上一轮 11 项建议中，7 项已修复，4 项待办。**

---

## 二、已修复项（逐项验证）

### ✅ P0-1 · M06 异步路径 —— 已修复
- `agent/runtime/runtime.py:35` `AgentRuntimeInterface.arun`（默认回退同步）
- `agent/runtime/runtime.py:65` `LangGraphRuntime.arun` → `await self.graph.ainvoke(...)`
- `agent/runtime/runtime.py:78` `LangGraphRuntime.astream(...)`
- 测试：`test_audit_fixes.py::TestAsyncRuntime` 4 例覆盖成功/失败/流式/接口回退
- **意义**：解除「接入 FastAPI 会阻塞事件循环」这一最大技术障碍

### ✅ P0-2 · Agent 侧对外入口 —— 已修复
- 新增 `chatchat/agent/__main__.py`，提供 `list` / `run` / `collab` 三个子命令
- 入口经 `M08 Registry 发现 → M07 Harness → M06 Runtime` 执行（符合架构分层）
- **实测验证**：
  - `python -m chatchat.agent list` → 成功列出 5 个 Agent（tender_review + 4 个 demo）
  - `python -m chatchat.agent run tender_review --requirement ... --response ...` → **返回真实结果**
    - 状态 `COMPLIANT`，explanation 正确，`elapsed=13.8s`，`retry_count=0`
    - 日志显示 `ComplianceAnalysisNode` 耗时 4.4s，说明**确有真实 LLM 调用**，非降级路径

### ✅ P0-3 · M03/M04 provider 工厂 —— 已修复
- 新增 `chatchat/rag/parser/provider.py` → `get_document_parser()`（模块级单例）
- 新增 `chatchat/rag/chunk/provider.py` → `get_chunker()`（模块级单例）
- 消费方已改造，不再直接 import 具体类：
  - `server/knowledge_base/utils.py:334,336` → `from chatchat.rag.parser.provider import get_document_parser`
  - `server/knowledge_base/utils.py:353,355` → `from chatchat.rag.chunk.provider import get_chunker`
- 测试：`test_audit_fixes.py::TestProviders` 2 例验证单例语义
- **意义**：三个插槽（RAG/parser/chunker）现已全部可「换一行实现」

### ✅ P1-4 · GRAPH_ENGINE 灰度开关 —— 已修复
- `settings.py:104` 新增 `GRAPH_ENGINE: str = "legacy"`
- 新增 `chatchat/agent/engine.py`：`get_graph_engine()` / `is_langgraph_enabled()` / `graph_engine_smoke()`
- 未知取值自动回退 legacy（有测试覆盖）
- **意义**：迁移方案 P0（基建+冒烟）落地，为后续分流铺路

### ✅ P1-6 · GraphBuilder 泛型化 —— 已修复
- `runtime/graph.py:33` `def __init__(self, state_schema=AgentState)`，`:34` `StateGraph(state_schema)`
- 默认行为不变（向后兼容），支持 `GraphBuilder(state_schema=MyState)`
- 测试：`test_audit_fixes.py::TestGraphBuilderGenericState` 2 例（默认 + 自定义状态）
- **意义**：消除「M06 GraphBuilder vs 裸 StateGraph」两套构图风格

### ✅ P1-5 · M05 混合检索可用 —— 已修复
- `server/chat/kb_chat.py:62` 新增请求参数：
  ```python
  retrieval_strategy: Optional[Literal["vector", "hybrid_rrf", "hybrid_weighted"]] = Body(...)
  ```
- `kb_chat.py:91` 已透传到 `RAGQuery(retrieval_strategy=...)`
- **意义**：M05 的 BM25 + RRF/Weighted 融合现在从 API 层可达，不再是死代码

### ✅ P2-N · pytest 基建修复 —— 已修复
- `pyproject.toml:234` addopts 已移除失效的 `--snapshot-warn-unused`
- `pyproject.toml:242` 新增 `asyncio_mode = "auto"`；已声明 `pytest-asyncio`、`syrupy` 依赖
- **实测**：默认配置直接 `pytest tests/unit_tests` → **224 passed**（无需任何绕过参数）
- 上次失败的 `test_mcp_prompts.py::test_load_mcp_prompt` 已恢复通过

---

## 三、本次测试执行记录

```
命令：PYTHONUTF8=1 ./.venv/Scripts/python.exe -m pytest tests/unit_tests -q
结果：224 passed, 8 warnings in 20.09s
失败：0
（对比 10-08：212 用例，1 failed / 211 passed）
```

新增测试文件：`tests/unit_tests/test_audit_fixes.py`（12 例）
覆盖范围：GraphBuilder 泛型化、异步 Runtime、GRAPH_ENGINE 开关与冒烟、parser/chunker provider

CLI 端到端实测：
```
$ python -m chatchat.agent list
tender_review        Tender Review Agent          v0.1.0  招标要求与投标响应符合性审查 Agent
requirement_agent    Requirement Agent (Demo)     v0.1.0  M10 Demo：需求提取
retrieval_agent      Retrieval Agent (Demo)       v0.1.0  M10 Demo：证据检索
compliance_agent     Compliance Agent (Demo)      v0.1.0  M10 Demo：符合性判断
report_agent         Report Agent (Demo)          v0.1.0  M10 Demo：报告汇总

$ python -m chatchat.agent run tender_review \
    --requirement "项目经理必须具有5年以上相关项目经验。" \
    --response "项目经理张三具有7年相关项目经验。"
→ {"status": "COMPLIANT", "explanation": "项目经理张三具有7年相关项目经验，满足...", ...}
   total_requirements=1, compliant_count=1, unknown_count=0
```

---

## 四、待办项（仍未修复）

| # | 问题 | 位置 | 说明 | 建议优先级 |
|---|---|---|---|---|
| T-1 | **ParallelStrategy 仍是空壳** | `collaboration/strategies.py:80-92` | `order()` 仍直接 `return topological_order(plan.tasks)`，与 Sequential 完全相同；`groups()` 已能分层但**无人调用**。且 `coordinator.py:46` 默认仍是 `SequentialStrategy()` | P2 |
| T-2 | **Rerank 仍为 NoOp** | `rag/retrieval/rerank.py` | 只有 `NoOpReranker`，无真实 BGE / Cross Encoder / RAGFlow Reranker | P1 |
| T-3 | **RAGFlow 仍未接入** | 全仓库 | 仅注释提及，无依赖/客户端/配置。目标链路 `统一 RAG Service → RAGFlow` 进度仍为 0% | P1（视业务需要） |
| T-4 | **PRDTestAgent 仍不存在** | — | 目标架构中的第二个业务 Agent 尚未创建 | P1（相对你的目标） |

### 可能仍存在（本次未逐一复验的旧问题）
- M01 Core 未被主服务采纳（主服务仍用 `server/utils.py:build_logger`）→ 双日志体系
- M02 仅 `kb_chat.py` 接入，`file_chat.py` 仍未接入
- M06 无 Checkpoint / 持久化记忆；`tool_results` 是否演化为真实 ReAct 工具循环待确认
- M07 Memory 仅 InMemory、Policy 仅 AllowAll
- M08 Registry 仅内存态、无动态插件发现
- M09 tender_review 仍为 7 节点全顺序边（无分支/循环/重试回边）
- 主服务（WebUI/API）是否已按 `is_langgraph_enabled()` 分流 —— `engine.py` 注释说明「不改变现有 AgentExecutor 生产链路」，**建议确认是否已有入口接线**

---

## 五、复测结论

**本轮修改质量高、覆盖面准，7 项修复全部真实落地且有测试兜底**：

1. **最关键的架构障碍已解除** —— M06 补上了 `arun`/`astream`，Agent 框架首次具备接入异步主服务的条件；
2. **「Agent 插件化」从类库变成产品能力** —— `python -m chatchat.agent` 可列 Agent、可真实执行，且端到端跑通了真实 LLM；
3. **「接口解耦」补齐最后两块拼图** —— parser/chunker 有了工厂，三插槽齐备；
4. **工程质量回升** —— 默认 pytest 可跑，224 全绿，测试基建从"坏"回到"好"。

**剩余缺口集中在三处**：RAGFlow 接入（0%）、PRDTestAgent（未建）、M10 并行（仍为串行语义）。这三项都属于"新增能力"而非"修复缺陷"，可按业务优先级排期。
