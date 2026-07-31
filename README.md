# My-Easy-Local-RAG

基于开源项目 [AllAboutAI-YT/easy-local-rag](https://github.com/AllAboutAI-YT/easy-local-rag) 二次开发的全本地检索增强生成工具，针对中文场景重构优化，解决原版代码冗余、中文适配差、功能简陋、配置失效等问题。

## 目录

1. 项目简介
2. 原版存在缺陷 & 本版本核心优化
3. 项目结构
4. 快速部署使用
5. 配置参数调优指南
6. 完整功能说明
7. 许可证说明
8. 致谢
   
   ## 1. 项目简介
   
   本项目衍生自 `AllAboutAI-YT/easy-local-rag`，一款基于 Ollama 的离线本地 RAG 系统。
   原版仅支持少量文件格式、无缓存、代码重复、对中文文档兼容性差；本版本进行底层重构，新增混合检索、中文分词优化、多格式文档解析、向量缓存、文档管理、对话导出、标准化配置等全套能力，完全适配中文知识库使用。
   基础技术栈：Python 3.13 + Ollama + PyTorch
   新增依赖：jieba、rank_bm25、python-docx、openpyxl、numpy
   
   ## 2. 原版存在缺陷 & 本版本核心优化
   
   ### 原版原生问题
9. `config.yaml` 配置文件未生效，全部参数硬编码
10. 每次启动全量重算向量，启动速度极慢
11. `localrag.py` / `localrag_no_rewrite.py` 90% 代码冗余重复
12. 文档仅支持 PDF/TXT/JSON，无 Word/Excel/Markdown 等常用格式
13. 分块规则仅适配英文标点，中文切分效果差
14. 无法删除、清空、查看已上传文档，文档管理缺失
15. 仅单一向量相似度检索，无关键词检索，中文专有名词召回弱
16. 无重排序机制，检索结果相关性低
17. 无对话记录导出功能
18. 上下文仅简单文本拼接，不标注文档来源，溯源困难
    
    ### 二次开发新增/重构
    
    #### 架构层面
- 抽取公共逻辑至 `core.py`，消除代码重复，统一封装配置、分块、检索、文档管理工具类
- 合并双 RAG 对话文件，通过命令行参数开关查询重写，精简工程文件
- 实现向量嵌入缓存机制，文档无修改时直接加载缓存，秒级启动
- 新增文档注册表 `vault_registry.json`，记录每个文本块对应的原始文件，支持精准删除
  
  #### 检索能力升级（中文核心优化）
- 引入 jieba 中文分词，实现 BM25 关键词检索
- 混合检索方案：BM25关键词 + 向量语义检索，RRF算法融合重排序
- 可自由调整两种检索权重，适配中文技术文档、通用文章、英文资料等不同场景
- 结构化上下文输出，每条检索片段携带编号与原始文档来源，便于溯源
  
  #### 文档上传与管理
- 支持8种主流格式：PDF / TXT / JSON / MD / DOCX / HTML / CSV / XLSX
- 自动清洗各格式冗余内容：Markdown标签、HTML脚本样式、表格结构化转文本
- 双运行模式：GUI可视化上传 + CLI命令行管理
- 支持查看文档列表、单文件删除、一键清空全部知识库、重复文件自动覆盖
  
  #### 对话交互优化
- 完整读取 `config.yaml` 统一管理模型、分块、检索参数
- 对话记录支持导出 Markdown / JSON 文件存档
- 多重异常防御：空查询、空向量、超长上下文、Windows中文编码崩溃自动处理
- 命令行灵活传参：切换模型、关闭查询重写、强制重建向量缓存
  
  ## 3. 项目完整结构
  
  ```
  easy-local-rag-main/
  ├── core.py # 公共核心模块（配置/分块/混合检索/文档管理）
  ├── upload.py # 文档上传、知识库管理（GUI + CLI双模式）
  ├── localrag.py # RAG对话主程序（合并原版两套对话逻辑）
  ├── config.yaml # 全局标准化配置文件
  ├── vault.txt # 文档分块原始文本存储
  ├── vault_registry.json # 文档来源注册表，用于删除溯源
  ├── vault_embeddings_cache.json # 向量嵌入缓存文件
  ├── requirements.txt # 完整依赖清单
  ├── .env # 环境变量配置
  # 已移除原版冗余文件
  # localrag_no_rewrite.py
  ```
  
  ## 4. 快速部署使用
  
  ### 4.1 环境准备
1. 克隆本项目
   
   ```bash
   git clone https://github.com/你的用户名/你的仓库名.git
   cd 你的仓库名
   ```
2. 安装全部依赖
   
   ```bash
   pip install -r requirements.txt
   ```
3. 安装并启动 Ollama，拉取所需模型
   
   ```bash
   # 大语言模型（可自行替换）
   ollama pull qwen3.6
   # 嵌入向量模型
   ollama pull nomic-embed-text
   ```
   
   ### 4.2 文档上传管理
   
   ```bash
   # 启动可视化GUI上传窗口
   python upload.py
   # 命令行直接上传单文件
   python upload.py 知识库文档.pdf
   # 查看已上传全部文档
   python upload.py --list
   # 删除指定文档
   python upload.py --remove 知识库文档.pdf
   # 一键清空全部知识库
   python upload.py --clear
   ```
   
   ### 4.3 启动RAG问答对话
   
   ```bash
   # 默认开启查询重写
   python localrag.py
   # 关闭查询重写
   python localrag.py --no-rewrite
   # 指定使用其他Ollama模型
   python localrag.py --model llama3
   # 对话结束后导出记录为Markdown
   python localrag.py --export chat_record.md
   # 强制重新生成向量缓存
   python localrag.py --rebuild-cache
   ```
   
   对话内输入 `quit` 即可退出程序。
   
   ## 5. 配置参数调优指南（config.yaml）
   
   ```yaml
   # 向量嵌入模型
   embedding_model: "nomic-embed-text"
   # 文本分块参数
   chunk_size: 300
   chunk_overlap: 80
   # 混合检索权重配置
   retrieval:
   bm25_weight: 0.6
   vector_weight: 0.4
   bm25_top_k: 10
   vector_top_k: 10
   final_top_k: 5
   # 存储文件路径
   registry_file: "vault_registry.json"
   embeddings_cache_file: "vault_embeddings_cache.json"
   ```
   
   ### 场景化参数推荐
4. 中文技术文档/带大量专有名词
   BM25权重0.6~0.7，向量权重0.3~0.4
5. 英文通用文章
   BM25权重0.3~0.4，向量权重0.6~0.7
6. FAQ、短问答文档
   chunk_size=200，overlap=50
7. 长篇书籍、完整技术手册
   chunk_size=500，overlap=100
   
   ## 6. 数据流说明
   
   ```
   用户上传文档
   ↓
   upload.py 解析多格式文件 → core.chunk_text() 中文优化分块
   ↓
   写入 vault.txt 文本库 + vault_registry.json 来源记录
   ↓
   用户输入问题启动对话
   ↓
   可选：查询重写优化模糊提问
   ↓
   core.hybrid_search() 混合检索
   ├─ jieba分词 BM25关键词检索
   ├─ Ollama向量语义相似度检索
   └─ RRF融合重排序，筛选top-k相关片段
   ↓
   结构化格式化上下文（编号+文档来源）
   ↓
   调用Ollama大模型生成回答
   ↓
   可选导出完整对话记录
   ```
   
   ## 7. 许可证说明
   
   ### 版权声明
8. 本项目基础代码、原始逻辑版权归原作者 AllAboutAI-YT 所有，原项目仓库：https://github.com/AllAboutAI-YT/easy-local-rag
9. 本项目所有新增、重构、优化代码版权归本二次开发作者所有。
   
   ### 协议约束
   
   本项目完全遵循原开源仓库许可证协议，根目录保留原版 LICENSE 文件，衍生项目整体沿用原协议约束。
- 禁止删除、修改、替换根目录原始 LICENSE 文件；
- 所有原始代码文件保留原作者版权声明；
- 商业使用、分发本项目必须同时遵守原开源协议全部条款。
  
  ## 8. 致谢
  
  非常感谢原作者 AllAboutAI-YT 开源 easy-local-rag 项目，为本中文增强版本地RAG系统提供完整底层原型与开发基础。
  原作者配套教程：
- https://www.youtube.com/watch?v=Oe-7dGDyzPM
- https://www.youtube.com/watch?v=vFGng_3hDRk
  原作者频道：https://www.youtube.com/c/AllAboutAI
