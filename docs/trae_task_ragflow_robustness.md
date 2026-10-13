# Trae 任务提示词：RAGFlow Parser 大批量场景健壮性优化

> 交付方式：把下面「任务提示词」整段复制给 Trae。

---

## 一、任务提示词（复制以下全部内容给 Trae）

### 背景

项目是 Langchain-Chatchat 的二次开发分支，位于 `libs/chatchat-server/`。  
已完成 RAGFlow DeepDoc 解析器接入（接法二：RAGFlow 只做解析、数据存本地），  
核心实现文件：`chatchat/rag/parser/service_ragflow.py`。

当前在**大批量上传文档**场景下暴露三个问题：

1. **单文件解析超时风险**：`RAGFLOW_PARSE_TIMEOUT`（运行值已调为 1800s，但代码默认值仍是 600s），
   大 PDF（几百页）解析可能超过此值 → 任务失败，且临时 dataset 残留无法回收。
2. **清理不彻底**：`parse()` 的 `finally` 在「进程被杀 / httpx 请求永久挂起」等情况下  
   可能不执行，导致 RAGFlow 侧残留 `cc_temp_parse_*` 临时库。且系统**没有任何兜底清理机制**。
3. **无并发控制**：批量上传时 `files2docs_in_thread` 用 `ThreadPoolExecutor()`（默认  
   max_workers = min(32, cpu+4)）无脑并发调用 RAGFlow。但 **RAGFlow ingestor 只有 1 个  
   worker**（日志：`Worker pool started with 1 workers`），并发请求只会在 RAGFlow 侧排队，  
   且大量任务可能同时撞上超时集体失败。

### 目标

**在不改变「接法二」架构、不破坏现有 237 个单测、默认行为向后兼容的前提下**，  
提升 RAGFlow Parser 在大批量场景下的健壮性。

### 需要实现的功能

#### 任务 1：可配置的并发闸门（Semaphore）

在 `RAGFlowDocumentParser` 中增加一个**进程级信号量**，限制同时进行的 RAGFlow  
解析会话数（同一时刻最多 N 个 temp dataset 在 RAGFlow 侧排队）。

- 新增配置项 `RAGFLOW_MAX_CONCURRENT`（默认值 `2`），加在  
  `chatchat/settings.py` 的 `BasicSettings` 中（紧邻其他 `RAGFLOW_*` 字段），  
  并在 `basic_settings.yaml` 模板中加注释说明。
- 在 `parse()` 的整套「建库→上传→解析→拉取→清理」流程外层加信号量保护，  
  **必须保证 `finally` 在信号量 release 之前执行**（用 try/finally 或 with 语句）。
- 信号量必须是**模块级/类级单例**，不能每个实例各持一个（否则多线程下失效）。
- 注意：`parse()` 是同步阻塞方法，多个线程调用会各自阻塞等待信号量，  
  这是预期行为。

#### 任务 2：临时 dataset 兜底清理（孤儿清理）

新增一个**孤儿临时库清理方法**，用于回收异常残留：

- 方法签名建议：`RAGFlowDocumentParser.cleanup_orphan_datasets(self, older_than_seconds: int = 3600) -> int`
- 实现逻辑：
  1. `GET /api/v1/datasets?page_size=1024` 拉取所有 dataset；
  2. 筛选 `name` 以 `self.temp_dataset_prefix` 开头的；
  3. 通过 `create_time`（毫秒时间戳）判断是否超过 `older_than_seconds`；
  4. 对超龄的批量 `DELETE /api/v1/datasets`（body: `{"ids": [...]}`）；
  5. 返回清理数量，并逐条打 INFO 日志。
- **安全要求**：只删以 `temp_dataset_prefix` 开头的库，绝不碰其他 dataset。  
  如果 RAGFlow 返回的 dataset 字段里没有 `create_time`，则退化为「只报告不删除」并打 WARNING。
- 在 `chatchat/cli.py` 增加一个子命令（或在现有 CLI 中挂载），例如：  
  `python -m chatchat.cli ragflow-cleanup --older-than 3600`  
  使其可被手动/定时调用。**注意本项目已有 `python -m chatchat.agent` 的 CLI 范式，请参考其写法保持风格一致。**

#### 任务 3：超时与轮询参数调优（默认值 + 代码健壮性）

> 说明：这两个参数的**运行时值已由人工调好**（`POLL_INTERVAL: 0.5`、`TIMEOUT: 1800.0`）。
> 本任务要求 Trae 把**代码内置默认值**与**模板注释**同步，并加固相关逻辑。

**3.1 修改代码默认值**（`chatchat/settings.py` 的 `BasicSettings`）

```python
RAGFLOW_PARSE_POLL_INTERVAL: float = 0.5    # 原 2.0 → 改 0.5
RAGFLOW_PARSE_TIMEOUT: float = 1800.0       # 原 600.0 → 改 1800.0
```

理由写进代码注释：轮询间隔调小可减少「解析已完成但仍空转等待」的时间（本地部署
轮询开销可忽略）；超时放宽避免大 PDF 被误判为失败、进而残留临时库。

**3.2 同步 `basic_settings.yaml` 模板注释**

两个字段都带中文说明，明确写出「大批量场景建议值」：
- `RAGFLOW_PARSE_POLL_INTERVAL`：注明「本地部署轮询无压力，大批量建议 0.5」
- `RAGFLOW_PARSE_TIMEOUT`：注明「大 PDF 解析可能超 10 分钟，大批量/大文件建议 1800」

**3.3 加固 `_wait_until_done` 的边界处理**

- 轮询间隔从 2.0 降到 0.5 后，**总轮询次数会翻 4 倍**。请确认循环内没有累积型开销
  （如每次轮询新建 `httpx.Client`）——`client` 应复用 `parse()` 传入的实例。
- 增加**连续失败容忍**：单次轮询抛网络异常（`httpx.TransportError`）时不要立即失败，
  记录 WARNING 后继续重试，直到 `parse_timeout` 耗尽。建议容忍连续 3 次失败。
- 超时抛 `ServiceError` 时，错误信息**必须包含已等待时长与实际超时配置值**，
  便于排查（例：`RAGFlow 解析超时（已等待 1800.0s / 上限 1800.0s）: <doc_id>`）。

**3.4 补充测试**

- 断言 `_wait_until_done` 在轮询异常时会重试（mock 前 2 次抛 `httpx.ConnectError`、
  第 3 次返回 DONE，断言最终成功）。
- 断言超时错误信息中包含超时配置值。

#### 任务 4：补充单元测试

在 `tests/unit_tests/` 下新增测试（沿用现有 pytest + `asyncio_mode="auto"` 风格）：

- 信号量生效：mock `httpx.Client`，并发调用 `parse()`，断言「同时进入临界区」的  
  数量不超过 `RAGFLOW_MAX_CONCURRENT`。
- 孤儿清理：mock 返回 3 个 dataset（2 个是 `cc_temp_parse_*` 且超龄、1 个是其他名字），  
  断言只删那 2 个。
- 清理保护：mock 返回的 dataset 全部不以 prefix 开头，断言**一个都不删**。
- 超时路径：mock 轮询永远返回 `progress=0.5`，断言抛 `ServiceError` 且  
  `_delete_dataset` 被调用（可用 unittest.mock 记录调用）。

### 硬性约束（务必遵守）

1. **不改变「接法二」架构**：RAGFlow 仍只做解析，解析完即删临时库，数据存本地。  
   不得改为持久 dataset、不得引入 RAGFlow 检索能力。
2. **默认行为向后兼容**：
   - `RAGFLOW_PARSER_ENABLED` 默认仍为 `false`；
   - 新增的 `RAGFLOW_MAX_CONCURRENT` 默认 `2`，但**当 `RAGFLOW_PARSER_ENABLED=false`  
     时整条链路完全不触发**，行为与改造前字节级一致。
3. **不引入重型新依赖**：只用 `httpx`（已有）、`threading`（标准库）、`time`、`uuid`。  
   不要引入 `celery`、`redis` 等。
4. **不影响 `CurrentDocumentParser`**：它仍是默认实现与回退实现。
5. **所有新增配置项需同步更新**：`chatchat/settings.py` 的 `BasicSettings` 字段 +  
   `basic_settings.yaml` 模板（含中文注释）。
6. **测试必须全绿**：改完后运行  
   `PYTHONUTF8=1 ./.venv/Scripts/python.exe -m pytest tests/unit_tests -q`，  
   现有 237 个测试 + 新增测试全部通过。
7. **代码注释用中文**，风格与 `service_ragflow.py` 现有注释保持一致（说明「为什么」而非「做什么」）。

### 交付物

1. 修改后的 `chatchat/rag/parser/service_ragflow.py`
2. 修改后的 `chatchat/settings.py`（新增 `RAGFLOW_MAX_CONCURRENT`）
3. 修改后的 `basic_settings.yaml`（新增配置项 + 注释）
4. CLI 孤儿清理命令（改 `chatchat/cli.py` 或新建入口）
5. 新增单元测试文件
6. **一份简短的改动说明**（放在回复里即可）：每个任务改了什么、为什么这么改、  
   测试结果（多少 passed）

### 参考信息（当前环境实际值，供你写代码时参考）

- RAGFlow 版本：`v1.0.0-rc1`，API 路径为**复数** `/api/v1/datasets/...`
- RAGFlow ingestor 单 worker（这是并发闸门存在的根本原因）
- 实测：一份 145KB PDF 解析约 10~25 秒
- 临时库命名：`cc_temp_parse_<12位uuid>`
- 相关配置项（`basic_settings.yaml`）：
  ```yaml
  RAGFLOW_PARSER_ENABLED: true
  RAGFLOW_BASE_URL: "http://127.0.0.1:9380"
  RAGFLOW_API_KEY: "ragflow-..."
  RAGFLOW_PARSE_POLL_INTERVAL: 0.5
  RAGFLOW_PARSE_TIMEOUT: 1800.0
  RAGFLOW_TEMP_DATASET_PREFIX: "cc_temp_parse_"
  ```

---

## 二、给用户的自查清单（Trae 交付后，用来验收）

Trae 改完后，逐条核对：

| # | 验收项                                          | 怎么验                                                          |
| - | -------------------------------------------- | ------------------------------------------------------------ |
| 1 | `RAGFLOW_MAX_CONCURRENT` 已加到 settings + yaml | `grep RAGFLOW_MAX_CONCURRENT basic_settings.yaml`            |
| 2 | 信号量是类级/模块级单例                                 | 读代码，确认不是 `self.__init__` 里创建的                                |
| 3 | 孤儿清理只删前缀匹配的库                                 | 读代码 + 跑对应单测                                                  |
| 4 | 单测全绿                                         | 跑 pytest，看 passed 数量（应 ≥ 237）                                |
| 5 | 默认行为没变                                       | 把 `RAGFLOW_PARSER_ENABLED` 设回 false，跑一次上传，耗时应在 2 秒级别（走本地）    |
| 6 | 孤儿清理命令可用                                     | 跑 `python -m chatchat.cli ragflow-cleanup --older-than 3600` |

**验收后提醒**：改动要让服务**重启**才生效，且配置要改  
`C:\Users\Lenovo\chatchat\basic_settings.yaml`（**运行时配置**），  
不是源码目录那个（这个坑今天已经踩过）。
