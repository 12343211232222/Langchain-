# RAGFlow 部署工作区（本文件用于记录部署决策与命令，非自动化脚本）

## 目标
只为 Langchain-Chatchat 的 `RAGFlowDocumentParser` 提供**解析**能力。
不需要 RAGFlow 的检索/知识库管理功能，所以可以精简部署。

## 硬件约束（实测 2026-10-12）
- CPU 16 核 / 24 线程           → 充足
- 内存 15.8 GB（可用 1.8 GB）   → 瓶颈，必须控制占用
- 磁盘可用 52.3 GB              → 勉强，镜像+数据约需 20-30 GB
- Docker 29.8.0 / Compose 5.5.1 → 已安装，引擎需手动启动

## 官方要求
- CPU ≥ 4 核 / RAM ≥ 16 GB / Disk ≥ 50 GB

## 三种部署形态（按资源占用从低到高）

### A. 精简部署（推荐先试）
- 向量库用 `infinity`（RAGFlow 内置，比 ES 省内存）
- 关闭不用的用途：不需要 ES 就能跑解析
- 预期占用：ragflow-server ~2G + infinity ~1G + mysql ~1G + redis ~0.3G + minio ~0.3G ≈ 5G
- 风险：仍然吃紧，需要先腾出内存

### B. 云端 RAGFlow（零本地占用）
- 使用 RAGFlow 官方在线服务
- 只需 RAGFLOW_BASE_URL + RAGFLOW_API_KEY
- 约束：需联网、需账号

### C. 扩容机器 / 部署到服务器
- 内存 ≥ 32 GB，长期最稳

## 关键路径
RAGFLOW_HOME = C:\Users\Lenovo\ragflow   （建议，避免中文路径）

## 部署命令（待 Docker 引擎启动后执行）
```bash
# 1) 克隆（浅克隆即可）
cd /c/Users/Lenovo
git clone --depth 1 https://github.com/infiniflow/ragflow.git
cd ragflow/docker

# 2) 选择精简版 compose（无 ES，用 infinity）
#    RAGFlow 官方提供 docker-compose-base.yml + 各 slim 变体
#    使用 docker-compose.yml 的 slim 版本

# 3) 启动
docker compose -f docker-compose.yml up -d

# 4) 查看状态
docker compose ps
docker logs -f ragflow-server
```

## 启动后验收
1. 浏览器打开 http://localhost:80 （或配置的端口）
2. 注册账号 → 右上角头像 → API → 生成 API Key
3. 把 API Key 填入 Langchain-Chatchat 的 basic_settings.yaml
4. 运行：`python scripts/ragflow_parser_smoke.py 某个.pdf`

## 注意
- RAGFlow 的 API Key 与登录密码不同，必须在「API」页面单独生成
- 端口 80 可能被占用，可在 .env 中改 `SVR_HTTP_PORT`
- 镜像约 15-20 GB，首次拉取耗时长
