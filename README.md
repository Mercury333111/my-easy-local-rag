# My-Easy-Local-RAG

基于开源项目 [AllAboutAI-YT/easy-local-rag](https://github.com/AllAboutAI-YT/easy-local-rag) 二次开发的全本地检索增强生成工具，针对中文场景重构优化，解决原版代码冗余、中文适配差、功能简陋、配置失效等问题。

## 目录

1. [项目简介](#1-项目简介)
2. [原版缺陷 & 核心优化](#2-原版缺陷--核心优化)
3. [项目结构](#3-项目结构)
4. [快速部署](#4-快速部署)
5. [使用说明](#5-使用说明)
6. [配置调优](#6-配置调优)
7. [数据流](#7-数据流)
8. [许可证](#8-许可证)
9. [致谢](#9-致谢)

---

## 1. 项目简介

本项目衍生自 `AllAboutAI-YT/easy-local-rag`，一款基于 Ollama 的离线本地 RAG 系统。

原版仅支持少量文件格式、无缓存、代码重复、对中文文档兼容性差；本版本进行底层重构，新增混合检索、中文分词优化、多格式文档解析、向量缓存、文档管理、对话导出、HTTP API、延迟指标、检索评估等全套能力。

**技术栈：** Python 3.10+ / Ollama / PyTorch / FastAPI

**新增依赖：** jieba、rank_bm25、python-docx、openpyxl、numpy、fastapi、uvicorn

---

## 2. 原版缺陷 & 核心优化

### 原版问题

| # | 问题 |
|---|------|
| 1 | `config.yaml` 配置未生效，参数全部硬编码 |
| 2 | 每次启动全量重算向量，启动极慢 |
| 3 | `localrag.py` / `localrag_no_rewrite.py` 90% 代码重复 |
| 4 | 仅支持 PDF/TXT/JSON，无 Word/Excel/Markdown |
| 5 | 分块仅适配英文标点，中文切分效果差 |
| 6 | 无法删除/查看已上传文档 |
| 7 | 仅向量检索，无关键词匹配，中文专有名词召回弱 |
| 8 | 无对话导出、无来源溯源 |

### 二次开发优化

**架构层面**
- 抽取 `core.py` 公共模块，消除代码重复
- 合并双对话文件，CLI 参数开关查询重写
- 向量嵌入缓存机制，秒级启动
- 文档注册表 `vault_registry.json`，支持精准删除和来源溯源

**检索能力**
- jieba 中文分词 + BM25 关键词检索
- 混合检索：BM25 + 向量语义，RRF 算法融合重排序
- 权重可调，适配不同场景
- 结构化上下文输出（编号 + 来源）

**文档管理**
- 支持 8 种格式：PDF / TXT / JSON / MD / DOCX / HTML / CSV / XLSX
- GUI + CLI 双模式上传
- 查看列表、单文件删除、一键清空、重复覆盖

**工程化能力（新增）**
- FastAPI HTTP API 服务，自带 Swagger 文档
- 延迟指标埋点（per-stage timing：query_rewrite / bm25 / vector / rrf / llm_generate）
- 检索质量评估脚本（Recall@K / MRR）

---

## 3. 项目结构

```
easy-local-rag-main/
├── core.py                        # 公共核心模块（配置/分块/混合检索/指标）
├── localrag.py                    # RAG 对话主程序（CLI）
├── upload.py                      # 文档上传与知识库管理（GUI + CLI）
├── api_server.py                  # FastAPI HTTP API 服务
├── evaluate.py                    # 检索质量评估脚本
├── config.yaml                    # 全局配置文件
├── vault.txt                      # 文档分块文本存储
├── vault_registry.json            # 文档来源注册表
├── vault_embeddings_cache.json    # 向量嵌入缓存
├── test_queries.json              # 评估测试集（需自行编写）
├── requirements.txt               # 依赖清单
└── .env                           # 环境变量
```

---

## 4. 快速部署

### 4.1 环境准备

```bash
git clone https://github.com/你的用户名/你的仓库名.git
cd 你的仓库名
pip install -r requirements.txt
```

### 4.2 安装 Ollama 并拉取模型

```bash
ollama pull qwen3.6          # 大语言模型（可替换）
ollama pull nomic-embed-text  # 嵌入向量模型
```

### 4.3 上传文档

```bash
python upload.py                        # GUI 上传窗口
python upload.py 知识库文档.pdf           # 命令行上传
python upload.py --list                 # 查看已上传文档
python upload.py --remove 知识库文档.pdf  # 删除指定文档
python upload.py --clear                # 清空全部
```

### 4.4 开始对话

```bash
python localrag.py                      # 默认模式
python localrag.py --no-rewrite         # 关闭查询重写
python localrag.py --model llama3       # 指定模型
python localrag.py --export chat.md     # 导出对话
python localrag.py --metrics            # 显示延迟指标
python localrag.py --metrics metrics.json  # 导出指标到文件
```

---

## 5. 使用说明

### 5.1 CLI 对话 (`localrag.py`)

| 参数 | 说明 |
|------|------|
| `--model MODEL` | 指定 Ollama 模型 |
| `--no-rewrite` | 关闭查询重写 |
| `--export FILE` | 导出对话（.json / .md） |
| `--rebuild-cache` | 强制重建向量缓存 |
| `--metrics [FILE]` | 启用延迟指标，可选导出 JSON |

对话内输入 `quit` / `exit` / `q` 退出。

### 5.2 HTTP API 服务 (`api_server.py`)

```bash
python api_server.py                    # 默认 127.0.0.1:8000
python api_server.py --port 8080        # 自定义端口
python api_server.py --host 0.0.0.0     # 监听所有接口
```

启动后访问 http://localhost:8000/docs 查看交互式 Swagger 文档。

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 + 系统状态 |
| `/query` | POST | 提问（返回 answer + sources + latency_ms） |
| `/upload` | POST | 上传文档（multipart/form-data） |
| `/documents` | GET | 列出已索引文档 |
| `/documents/{name}` | DELETE | 删除指定文档 |
| `/metrics` | GET | 返回延迟统计 |

**示例请求：**

```bash
# 提问
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "斗地主的基本规则是什么？"}'

# 上传文档
curl -X POST http://localhost:8000/upload \
  -F "file=@文档.pdf"

# 查看文档列表
curl http://localhost:8000/documents
```

**响应示例 (`/query`)：**

```json
{
  "answer": "斗地主是一种三人扑克牌游戏...",
  "sources": [
    {"text": "斗地主使用54张牌...", "score": 0.0833, "index": 0, "source": "斗地主赛事说明.docx"}
  ],
  "latency_ms": {
    "query_rewrite_ms": 523.1,
    "retrieval_ms": 45.2,
    "llm_generate_ms": 1834.6,
    "total_ms": 2402.9
  }
}
```

### 5.3 检索评估 (`evaluate.py`)

```bash
# 生成测试集模板
python evaluate.py --generate-sample

# 编辑 test_queries.json，填入真实问题和期望命中的文档名
# [
#   {"query": "斗地主的叫牌规则", "relevant_sources": ["斗地主赛事说明.docx"]},
#   {"query": "春天是什么意思", "relevant_sources": ["斗地主赛事说明.docx"]}
# ]

# 运行评估
python evaluate.py

# 导出详细结果
python evaluate.py --output results.json
```

**输出示例：**

```
=== Evaluation Results ===
  Queries:      10
  Vault chunks: 44
  Documents:    1

  MRR:          0.833
  recall@1      0.700
  recall@3      0.900
```

### 5.4 延迟指标

CLI 对话模式下使用 `--metrics` 启用：

```
=== Latency Metrics ===
  query_rewrite       avg=  523.1ms  min=  312.0ms  max=  834.2ms  p50=  480.5ms  (n=5)
  bm25_search         avg=    3.2ms  min=    1.1ms  max=    8.4ms  p50=    2.8ms  (n=5)
  vector_search       avg=   42.1ms  min=   35.0ms  max=   58.3ms  p50=   40.2ms  (n=5)
  rrf_fusion          avg=    0.1ms  min=    0.1ms  max=    0.2ms  p50=    0.1ms  (n=5)
  context_format      avg=    0.3ms  min=    0.1ms  max=    0.8ms  p50=    0.2ms  (n=5)
  llm_generate        avg= 1834.6ms  min= 1200.0ms  max= 2500.1ms  p50= 1800.3ms  (n=5)
  total               avg= 2403.4ms  min= 1548.2ms  max= 3401.8ms  p50= 2324.0ms  (n=5)
```

---

## 6. 配置调优

`config.yaml` 完整参数：

```yaml
# 模型配置
ollama_model: "qwen3.6"
embedding_model: "nomic-embed-text"

# 分块参数
chunk_size: 300        # 每个 chunk 的目标字符数
chunk_overlap: 80      # chunk 之间的重叠字符数

# 混合检索配置
retrieval:
  bm25_weight: 0.6     # BM25 关键词检索权重
  vector_weight: 0.4   # 向量语义检索权重
  bm25_top_k: 10       # BM25 候选数量
  vector_top_k: 10     # 向量候选数量
  final_top_k: 5       # RRF 融合后最终返回数量

# 相似度阈值（低于此分数的向量检索结果被过滤）
similarity_threshold: 0.3

# Ollama API 配置
ollama_api:
  base_url: "http://localhost:11434/v1"
  api_key: "qwen3.6"
```

### 场景推荐

| 场景 | BM25 权重 | 向量权重 | chunk_size | overlap |
|------|-----------|----------|------------|---------|
| 中文技术文档/专有名词多 | 0.6~0.7 | 0.3~0.4 | 300 | 80 |
| 英文通用文章 | 0.3~0.4 | 0.6~0.7 | 400 | 100 |
| FAQ / 短问答 | 0.5 | 0.5 | 200 | 50 |
| 长篇书籍/技术手册 | 0.4 | 0.6 | 500 | 100 |

---

## 7. 数据流

```
用户上传文档
    ↓
upload.py 解析多格式文件 → core.chunk_text() 中文优化分块
    ↓
写入 vault.txt + vault_registry.json
    ↓
用户输入问题（CLI / HTTP API）
    ↓
可选：LLM 查询重写优化模糊提问
    ↓
core.hybrid_search() 混合检索
 ├─ jieba 分词 → BM25 关键词检索
 ├─ Ollama → 向量语义相似度检索
 └─ RRF 融合重排序 → top-k
    ↓
结构化上下文（编号 + 来源）
    ↓
Ollama LLM 生成回答
    ↓
返回结果（含延迟指标 / 可导出对话）
```

---

## 8. 许可证

### 版权

- 基础代码、原始逻辑：AllAboutAI-YT（[原项目](https://github.com/AllAboutAI-YT/easy-local-rag)）
- 所有新增、重构、优化代码：本二次开发作者

### 协议

完全遵循原开源仓库许可证协议，根目录保留原版 LICENSE 文件。

- 禁止删除/修改根目录 LICENSE 文件
- 原始代码文件保留原作者版权声明
- 商业使用/分发须遵守原协议全部条款

---

## 9. 致谢

感谢原作者 AllAboutAI-YT 开源 easy-local-rag 项目。

- 原作者教程：[视频1](https://www.youtube.com/watch?v=Oe-7dGDyzPM) | [视频2](https://www.youtube.com/watch?v=vFGng_3hDRk)
- 原作者频道：https://www.youtube.com/c/AllAboutAI
