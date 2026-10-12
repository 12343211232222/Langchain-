# Trae 开发任务书：接入 RAGFlow DeepDoc 作为 DocumentParser 实现

> 使用方式：把本文档全文（或分章节）粘贴给 Trae。  
> 编码前请先阅读「一、背景与约束」，它决定了实现的边界，不要自行扩大范围。

---

## 一、背景与约束

### 1.1 项目背景

项目：`Langchain-Chatchat`（自研改造版 0.3.1.3），服务端在 `libs/chatchat-server`。

项目有一个自研的里程碑体系（代码 docstring 中标注 M01–M10），其中与本次任务相关的三层抽象：

- **M03 Document Parser**：`chatchat/rag/parser/` — 把文件解析成 `Document` 列表
- **M04 Chunker**：`chatchat/rag/chunk/` — 把 `Document` 切分成 chunk
- **M02 RAG Service**：`chatchat/rag/` — 统一检索入口

三层都已做了接口抽象 + 工厂函数（provider），当前实现类是 `CurrentXxx`：

- `chatchat/rag/parser/interfaces.py` → `DocumentParser`（抽象）
- `chatchat/rag/parser/service.py` → `CurrentDocumentParser`（基于本地 document_loaders）
- `chatchat/rag/parser/provider.py` → `get_document_parser()`（模块级单例工厂）
- `chatchat/rag/chunk/*` 同构

### 1.2 本次任务的目标（严格限定）

**只替换 Parser 这一层**，让"文件 → Document"这一步改用 RAGFlow 的 DeepDoc 解析能力。

**明确不做（重要，不要越界）：**

- ❌ 不替换 Chunker（切分仍走 `CurrentChunker`）
- ❌ 不替换 embedding（仍用现有 `embedding-3`，智谱）
- ❌ 不替换检索（仍走现有向量库 FAISS/Milvus）
- ❌ 不修改 `kb_chat.py`、不上传文档到 RAGFlow 的持久 dataset
- ❌ 不引入 RAGFlow 作为检索引擎，不维护双份知识库
- ❌ **不改动现有 `CurrentDocumentParser` 的行为**（它必须保持可用，作为默认与回退）

一句话：RAGFlow 在本任务中只扮&#x6F14;**"一个把文件解析成结构化文本的函数"**，解析完就结束，不在它那边留存任何数据。

### 1.3 架构约束（必须遵守）

1. **依赖方向**：消费方（`chatchat/server/knowledge_base/utils.py`）必须只依赖 `DocumentParser` 接口与 `get_document_parser()` 工厂，**不得 import 任何 RAGFlow 具体实现**。
2. **不改接口签名**：`DocumentParser.parse(self, file_path: str, **kwargs) -> List[Document]` 保持不变。
3. **同步语义**：接口是同步的。RAGFlow 解析是异步的（需要轮询），因此实现内部要完成「提交 → 轮询 → 拉取 → 清理」的完整同步封装。
4. **默认可回退**：默认仍使用 `CurrentDocumentParser`。RAGFlow 未配置或调用失败时，必须能降级，不得让知识库摄取链路整体失败。
5. **不污染环境**：现有 `.venv` 是 Python 3.10 + 钉死的 `langchain 0.1.17` / `langchain-core 0.1.x`。**不得新增重型依赖**（onnxruntime / opencv / pandas 等一律不允许）。HTTP 调用请复用项目已有的 `httpx = 0.27.2`。
6. **不吞异常**：错误要抛出项目统一异常（`chatchat/core/exceptions.py` 中的 `ChatchatError` 子类，如 `ServiceError` / `ValidationError`）。

### 1.4 运行环境

- 工作目录：`libs/chatchat-server`
- 解释器：`.venv/Scripts/python.exe`（Python 3.10.11）
- **Windows 中文路径环境，所有 Python 命令必须前置 `PYTHONUTF8=1`**，否则会在加载 `.pth` 时报 `UnicodeDecodeError`
- 项目已配置 pytest，运行单测：`PYTHONUTF8=1 ./.venv/Scripts/python.exe -m pytest tests/unit_tests -q`

---

## 二、需要交付的产物

### 产物 1：`chatchat/rag/parser/service_ragflow.py`（新增）

实现 `RAGFlowDocumentParser(DocumentParser)`，内部流程：

```
parse(file_path)
  1. 校验文件存在（不存在抛 ResourceNotFoundError）
  2. 确保有一个"临时 dataset"（用于本次解析，用完即删）
  3. 上传文件到该 dataset           → POST /api/v1/datasets/{dataset_id}/documents
  4. 触发解析                       → POST /api/v1/datasets/{dataset_id}/chunks
  5. 轮询解析进度直到完成/超时      → GET  /api/v1/datasets/{dataset_id}/documents
  6. 拉取解析结果（chunks）         → GET  /api/v1/datasets/{dataset_id}/documents/{document_id}/chunks
  7. 转换为 List[langchain.docstore.document.Document]
  8. finally: 删除临时 dataset（保证不留数据）
```

**实现要点：**

- 用 `httpx.Client`，超时读 `Settings.basic_settings.HTTPX_DEFAULT_TIMEOUT`（默认 300s）
- 鉴权头：`Authorization: Bearer {api_key}`
- 轮询要有**最大等待时间**与**间隔**（建议间隔 2s，上限可配置，超时抛 `ServiceError`）
- 解析完成后拉到的 chunks，转换规则：
  - `Document.page_content` ← chunk 的 `content`（若有 `content_with_weight` 字段则优先用它）
  - `Document.metadata` 至少保留：`source`（原始文件路径，**必须与 `CurrentDocumentParser` 保持一致**，下游 `format_reference` 与 `metadata` 依赖它）、以及 `document_keyword` / `similarity`（若存在）
- **禁止**在此类中做切分（chunk 是切分单位，但 RAGFlow 返回的 chunk 直接当 Document 输出，后续仍会被 `CurrentChunker` 再处理一次——这是既定设计，不要"优化"掉）

### 产物 2：改造 `chatchat/rag/parser/provider.py`

保持 `get_document_parser()` 签名不变，改为按配置选择实现：

```python
def get_document_parser() -> DocumentParser:
    """获取当前 Document Parser（模块级单例）。"""
```

要求：

- 读取配置项 `Settings.basic_settings.RAGFLOW_PARSER_ENABLED`（布尔，默认 `False`）
- 为 `False`（默认）→ 返回 `CurrentDocumentParser()`（**保持现状行为完全不变**）
- 为 `True` → 返回 `RAGFlowDocumentParser()`
- 未知/配置缺失 → 回退 `CurrentDocumentParser()`，并打 warning 日志
- 保留单例语义（多次调用返回同一实例）

### 产物 3：`chatchat/settings.py` 新增配置项

在 `BasicSettings` 中新增（参照该文件既有字段风格，注意别破坏 pydantic 校验）：

```python
# RAGFlow 解析器开关（默认关闭，使用本地 document_loaders）
RAGFLOW_PARSER_ENABLED: bool = False
# RAGFlow 服务地址
RAGFLOW_BASE_URL: str = ""
# RAGFlow API Key
RAGFLOW_API_KEY: str = ""
# 解析轮询间隔（秒）
RAGFLOW_PARSE_POLL_INTERVAL: float = 2.0
# 解析最长等待时间（秒）
RAGFLOW_PARSE_TIMEOUT: float = 600.0
# 用于解析的临时 dataset 名称前缀
RAGFLOW_TEMP_DATASET_PREFIX: str = "cc_temp_parse_"
```

同时在 `libs/chatchat-server/basic_settings.yaml` 中补上同名配置项与中文注释（该文件是配置模板，用户会直接编辑）。

### 产物 4：`tests/unit_tests/test_ragflow_parser.py`（新增）

纯单元测试，**不得发起真实网络请求**（用 `unittest.mock` / `monkeypatch` 打桩 `httpx.Client`）。至少覆盖：

1. 文件不存在 → 抛 `ResourceNotFoundError`
2. 正常流程：mock 上传/解析/轮询/拉取 → 断言返回的 `Document` 列表，`page_content` 与 `metadata["source"]` 正确
3. 轮询超时 → 抛 `ServiceError`
4. 上游返回非 0 code（如 `{"code": 102, "message": ...}`）→ 抛 `ServiceError` 且错误信息包含原文
5. 异常路径下 **临时 dataset 仍被清理**（断言 delete 被调用）
6. provider 开关：`RAGFLOW_PARSER_ENABLED=False` → 返回 `CurrentDocumentParser`；`=True` → 返回 `RAGFlowDocumentParser`；未知值 → 回退
7. 单例语义：多次 `get_document_parser()` 返回同一对象

### 产物 5：更新 `chatchat/rag/parser/__init__.py`

在 `__all__` 中导出新类（保持既有导出不变，只做加法）。

---

## 三、RAGFlow API 参考（务必核对版本）

> ⚠️ **重要**：RAGFlow 不同版本接口参数名有差异（例如检索接口老版用 `datasets`、新版用 `dataset_ids`；上传/解析路径在老版本是单数 `/api/v1/dataset/...`，新版是复数 `/api/v1/datasets/...`）。**编码前请以你实际部署的 RAGFlow 版本的官方 API 文档为准**，不要照抄本文档的路径。本文档仅说明需要调用哪些能力。

需要的能力与对应接口（请按实际版本调整路径与字段）：

| 步骤          | 能力                              | 参考接口                                                               |
| ----------- | ------------------------------- | ------------------------------------------------------------------ |
| 建临时 dataset | 创建数据集                           | `POST /api/v1/datasets`                                            |
| 上传文件        | 多部分表单上传                         | `POST /api/v1/datasets/{dataset_id}/documents`                     |
| 触发解析        | 启动异步解析任务                        | `POST /api/v1/datasets/{dataset_id}/chunks`                        |
| 查询进度        | 轮询文档状态（看 `progress` / `run` 字段） | `GET /api/v1/datasets/{dataset_id}/documents`                      |
| 拉取结果        | 取 chunk 列表                      | `GET /api/v1/datasets/{dataset_id}/documents/{document_id}/chunks` |
| 清理          | 删除 dataset                      | `DELETE /api/v1/datasets`                                          |

**统一响应格式**（需处理）：

- 成功：`{"code": 0, "data": {...}}`
- 失败：`{"code": 102, "message": "xxx"}`
- 鉴权失败：HTTP 401 / `code: 401`

**鉴权**：`Authorization: Bearer {RAGFLOW_API_KEY}`

---

## 四、关键代码参考

### 4.1 现有接口（不要改）

```python
# chatchat/rag/parser/interfaces.py
class DocumentParser(ABC):
    @abstractmethod
    def parse(self, file_path: str, **kwargs) -> List[Document]:
        """解析文件为 Document 列表。"""
        raise NotImplementedError
```

### 4.2 现有实现（作为行为对齐的基准）

```python
# chatchat/rag/parser/service.py
class CurrentDocumentParser(DocumentParser):
    def parse(self, file_path: str, loader_kwargs: Optional[dict] = None, **kwargs) -> List[Document]:
        if not os.path.isfile(file_path):
            raise ResourceNotFoundError(f"文件不存在: {file_path}")
        ext = os.path.splitext(file_path)[-1].lower()
        loader_name = get_LoaderClass(ext)
        if loader_name is None:
            raise ValidationError(f"暂未支持的文件格式: {ext or '(无扩展名)'}")
        loader = get_loader(loader_name=loader_name, file_path=file_path, loader_kwargs=loader_kwargs)
        if isinstance(loader, TextLoader):
            loader.encoding = "utf8"
        return loader.load()
```

### 4.3 消费方（不改，仅作理解）

```python
# chatchat/server/knowledge_base/utils.py:331-339
def file2docs(self, refresh: bool = False):
    if self.docs is None or refresh:
        logger.info(f"{self.document_loader_name} used for {self.filepath}")
        from chatchat.rag.parser.provider import get_document_parser
        self.docs = get_document_parser().parse(
            self.filepath, loader_kwargs=self.loader_kwargs
        )
    return self.docs
```

### 4.4 项目统一异常（请复用）

```python
from chatchat.core.exceptions import (
    ChatchatError, ConfigurationError, ServiceError,
    ResourceNotFoundError, ValidationError,
)
```

---

## 五、验收标准

编码完成后，必须自行执行并确认以下全部通过：

```bash
cd libs/chatchat-server

# 1. 新增测试全绿
PYTHONUTF8=1 ./.venv/Scripts/python.exe -m pytest tests/unit_tests/test_ragflow_parser.py -q

# 2. 全量回归无破坏（当前基线：224 passed）
PYTHONUTF8=1 ./.venv/Scripts/python.exe -m pytest tests/unit_tests -q

# 3. 默认开关关闭时，行为与改造前完全一致
PYTHONUTF8=1 ./.venv/Scripts/python.exe -c "from chatchat.rag.parser.provider import get_document_parser; p = get_document_parser(); print(type(p).__name__)"
# 期望输出：CurrentDocumentParser

# 4. 依赖未被污染（不得新增 onnxruntime / opencv / pandas 等）
git diff pyproject.toml
```

**验收 checklist：**

- [ ] 全量测试仍为 **224 passed**（新增用例后总数应为 224 + 新增数，且 0 failed）
- [ ] 默认（`RAGFLOW_PARSER_ENABLED=False`）行为与改造前**完全一致**
- [ ] 消费方 `knowledge_base/utils.py` **未被修改**
- [ ] `pyproject.toml` 依赖**未新增重型包**
- [ ] 所有新代码有类型注解与中文 docstring，docstring 中标注所属里程碑（`M03`）
- [ ] 异常路径下临时 dataset 一定被清理（有测试覆盖）

---

## 六、实施步骤建议

1. 先读 `chatchat/rag/parser/` 下三个文件 + `chatchat/rag/chunk/provider.py`（作为工厂写法的参照）
2. 新增 `service_ragflow.py` 骨架（类 + `parse` 方法签名 + 异常处理框架），暂不填 HTTP 细节
3. 改 `provider.py` 支持开关切换
4. 改 `settings.py` + `basic_settings.yaml` 加配置项
5. 写 `test_ragflow_parser.py`（全部打桩，不联网）
6. 跑测试，修到全绿
7. 跑全量回归确认 224 + 新增 = 全部 passed

---

## 七、注意事项汇总

1. **不要扩大范围** —— 本次只动 Parser 层。看到 Chunker/Retrieval 的优化空间请记录但不要改。
2. **不要照抄接口路径** —— RAGFlow 版本差异大，以实际部署版本文档为准。
3. **不要新增依赖** —— 用现有 `httpx`。
4. **不要联网测试** —— 单测必须全部打桩。
5. **不要让默认行为变化** —— 开关默认关闭，改造后不配置就应表现得和现在一模一样。
6. **中文路径环境** —— 所有 Python 命令前置 `PYTHONUTF8=1`。
7. 若在实现中发现 RAGFlow 某接口无法满足"用完即删"的临时解析需求（例如必须依赖持久 dataset），**不要自行改变架构**，停下来输出问题说明与可选方案，交由人工决策。
