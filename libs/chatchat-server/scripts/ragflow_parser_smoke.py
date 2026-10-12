"""RAGFlow Document Parser 真实链路联调脚本（M03 接法二）。

用途：在不启动 Chatchat 主服务的前提下，单独验证 ``RAGFlowDocumentParser``
与真实 RAGFlow 服务的联通性，逐步骤打印诊断信息，便于快速定位
接口路径 / 返回字段 / 认证 等版本差异问题。

用法::

    # 1) 设置环境变量（或用命令行参数覆盖）
    set RAGFLOW_BASE_URL=http://127.0.0.1:9380
    set RAGFLOW_API_KEY=ragflow-xxxxxxxx

    # 2) 运行
    .venv\\Scripts\\python.exe scripts\\ragflow_parser_smoke.py path\\to\\sample.pdf

    # 只探活、不做完整解析
    .venv\\Scripts\\python.exe scripts\\ragflow_parser_smoke.py --probe-only

注意：本脚本会在 RAGFlow 上创建临时 dataset，正常结束后自动删除；
若中途异常退出，可能残留 ``cc_temp_parse_*`` 命名的 dataset，需手工清理。
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

# 允许以 `python scripts/xxx.py` 直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx  # noqa: E402

from chatchat.settings import Settings  # noqa: E402

OK = "[ OK ]"
FAIL = "[FAIL]"
INFO = "[INFO]"


def _mask(secret: str) -> str:
    """密钥脱敏显示。"""
    if not secret:
        return "(空)"
    if len(secret) <= 8:
        return secret[:2] + "*" * (len(secret) - 2)
    return f"{secret[:6]}...{secret[-4:]} (len={len(secret)})"


def print_config(base_url: str, api_key: str) -> None:
    cfg = Settings.basic_settings
    print("=" * 68)
    print("RAGFlow Parser 联调配置")
    print("=" * 68)
    print(f"  BASE_URL                : {base_url or '(空)'}")
    print(f"  API_KEY                 : {_mask(api_key)}")
    print(f"  PARSE_POLL_INTERVAL     : {cfg.RAGFLOW_PARSE_POLL_INTERVAL}s")
    print(f"  PARSE_TIMEOUT           : {cfg.RAGFLOW_PARSE_TIMEOUT}s")
    print(f"  TEMP_DATASET_PREFIX     : {cfg.RAGFLOW_TEMP_DATASET_PREFIX}")
    print(f"  RAGFLOW_PARSER_ENABLED  : {cfg.RAGFLOW_PARSER_ENABLED}")
    print("=" * 68)


def probe(base_url: str, api_key: str) -> bool:
    """探活：调用 list datasets，验证地址 / 认证 / 接口版本是否匹配。"""
    print(f"\n{INFO} 步骤 1/2：探活 RAGFlow 服务 ...")
    url = f"{base_url.rstrip('/')}/api/v1/datasets"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(url, headers=headers, params={"page": 1, "page_size": 1})
    except Exception as e:
        print(f"{FAIL} 无法连接 {base_url}: {type(e).__name__}: {e}")
        print("       排查：RAGFlow 是否已启动？端口是否正确？地址是否带 http:// 前缀？")
        return False

    print(f"       HTTP {resp.status_code}  (GET {url})")
    if resp.status_code == 401:
        print(f"{FAIL} 认证失败（401）。请检查 RAGFLOW_API_KEY 是否为 RAGFlow 界面里的 API Key")
        print("       注意：不是登录密码，需在 RAGFlow『API』页面单独生成。")
        return False
    if resp.status_code == 404:
        print(f"{FAIL} 接口 404。你部署的 RAGFlow 可能使用旧版单数路径")
        print("       旧版：/api/v1/dataset/...   新版：/api/v1/datasets/...")
        print("       需同步修改 chatchat/rag/parser/service_ragflow.py 中的路径。")
        return False

    try:
        payload = resp.json()
    except ValueError:
        print(f"{FAIL} 返回非 JSON（HTTP {resp.status_code}），响应前 200 字符：")
        print("       " + resp.text[:200].replace("\n", " "))
        return False

    code = payload.get("code")
    if code != 0:
        print(f"{FAIL} RAGFlow 业务错误 code={code}: {payload.get('message')}")
        return False

    data = payload.get("data")
    existing = len(data) if isinstance(data, list) else "?"
    print(f"{OK} 服务可达、认证通过、响应格式为 {{code, data}}（当前已有 {existing} 个 dataset）")
    return True


def parse_one(parser, file_path: str) -> bool:
    """执行一次完整解析，打印耗时与结果摘要。"""
    print(f"\n{INFO} 步骤 2/2：解析文件 {file_path}")
    if not os.path.isfile(file_path):
        print(f"{FAIL} 文件不存在：{file_path}")
        return False

    size_kb = os.path.getsize(file_path) / 1024
    print(f"       文件大小：{size_kb:.1f} KB")

    t0 = time.monotonic()
    try:
        docs = parser.parse(file_path)
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"{FAIL} 解析失败（耗时 {elapsed:.1f}s）：{type(e).__name__}: {e}")
        print("\n--- 完整堆栈 ---")
        traceback.print_exc()
        return False

    elapsed = time.monotonic() - t0
    print(f"{OK} 解析成功（耗时 {elapsed:.1f}s），返回 {len(docs)} 个 Document")

    if not docs:
        print(f"{FAIL} 返回为空。可能原因：文件为扫描件需配置 OCR 模型 / 解析器未匹配该文件类型")
        return False

    total_chars = sum(len(d.page_content) for d in docs)
    print(f"       总字符数：{total_chars}")
    print("\n--- 前 3 个 Document 预览 ---")
    for i, d in enumerate(docs[:3], 1):
        preview = d.page_content[:180].replace("\n", " ")
        print(f"  [{i}] metadata={d.metadata}")
        print(f"      content : {preview}{'...' if len(d.page_content) > 180 else ''}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="RAGFlow Document Parser 真实链路联调")
    ap.add_argument("file", nargs="?", help="待解析文件路径（.pdf/.docx/.txt 等）")
    ap.add_argument("--base-url", default=None, help="覆盖 RAGFLOW_BASE_URL")
    ap.add_argument("--api-key", default=None, help="覆盖 RAGFLOW_API_KEY")
    ap.add_argument("--probe-only", action="store_true", help="仅探活，不做解析")
    args = ap.parse_args()

    cfg = Settings.basic_settings
    base_url = args.base_url if args.base_url is not None else cfg.RAGFLOW_BASE_URL
    api_key = args.api_key if args.api_key is not None else cfg.RAGFLOW_API_KEY

    print_config(base_url, api_key)

    if not base_url or not api_key:
        print(f"\n{FAIL} 缺少 RAGFLOW_BASE_URL 或 RAGFLOW_API_KEY。")
        print("      两种设置方式：")
        print("      1) 编辑 libs/chatchat-server/basic_settings.yaml，填入 RAGFLOW_* 字段")
        print("      2) 运行时覆盖：--base-url http://127.0.0.1:9380 --api-key ragflow-xxx")
        return 2

    if not probe(base_url, api_key):
        return 1

    if args.probe_only:
        print(f"\n{OK} 探活通过（--probe-only，跳过解析）")
        return 0

    if not args.file:
        print(f"\n{INFO} 未提供待解析文件，跳过步骤 2。")
        print("      用法：python scripts/ragflow_parser_smoke.py path/to/sample.pdf")
        return 0

    # 直接构造解析器（绕过 provider 开关，便于联调）
    from chatchat.rag.parser.service_ragflow import RAGFlowDocumentParser

    parser = RAGFlowDocumentParser(base_url=base_url, api_key=api_key)
    ok = parse_one(parser, args.file)

    print("\n" + "=" * 68)
    print("联调结论：" + ("通过 ✅" if ok else "未通过 ❌"))
    print("=" * 68)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
