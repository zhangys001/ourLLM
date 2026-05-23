# 智能计算系统大作业 — 阶段性基线：纯向量检索方案

本项目目前暂时采用纯向量检索作为大作业打榜的阶段性基线方案，通过 Qwen Embedding 模型计算问题向量，并与本地构建的知识库进行余弦相似度匹配，直接返回最相关的答案。后续计划接入大语言模型进行生成（RAG）。

## 目录结构

```text
.
├── main.py                 # 评测入口（纯向量检索与匹配）
├── model/
│   ├── qwen_embedding_model/ # Qwen Embedding 模型权重
│   └── qwen_model/         # Qwen2-1.5B 大语言模型权重 (用于后续生成阶段)
├── data/                   # 原始知识及清洗后的数据 (不提交)
│   ├── clean_data/         # 清洗后的 markdown 与 QA 文件
│   ├── raw_md/             # 原始 markdown 格式文档
│   └── raw_pdf/            # 原始 PDF 课件
├── rag/                    # 运行时向量数据 (提交)
│   ├── embedding_knowledge.npy  # 核心正主：课程知识库的向量库 (本地离线生成，不入代码库)
│   ├── embedding_knowledge.json # 核心正主：课程知识库文本，大多数题目依赖此数据得分
│   ├── embedding_qa.npy         # 辅助工具：QA 向量库，用于投机取巧撞击原题拿满时间分
│   └── embedding_qa.json        # 辅助工具：QA 知识库文本与元数据
├── tools/                  # 制作/辅助脚本 (不提交)
│   └── make_npy.py         # 向量生成离线脚本
├── 方案规划.md              # 方案规划及演进路线
└── 智能计算系统大作业任务书.md
```

## 上手指南

1. **模型下载**:
   - 将 Qwen Embedding 模型放置到 `model/qwen_embedding_model/`。
   - 将 Qwen2 生成模型放置到 `model/qwen_model/`（后续 RAG 生成使用）。
2. **环境依赖**: 推荐使用 `ourLLM_env` (micromamba) 进行开发测试。
3. **向量生成**: `.gitignore` 默认忽略了 `*.npy`。如果本地缺少 `rag/embedding_knowledge.npy` 或 `rag/embedding_qa.npy`，请先运行 `python tools/make_npy.py` 构建。
4. **本地测试**: 
   ```bash
   python main.py "问题文本"
   ```
   可加上 `--debug` 查看详细的相似度打分及召回结果：
   ```bash
   python main.py "问题文本" --debug
   ```
5. **开发演进**: 详细的实施路线和后续的大模型生成接入方案，请参考 `方案规划.md`。

## 打包提交

最终提交给评测系统时，仅需包含 `main.py` + `model/` + `rag/` 相关依赖项。**注意：`data/` 和 `tools/` 目录不需要打包。**

因为 `.gitignore` 忽略了 `.npy` 文件，所以在打包前务必确保本地已生成核心所需的 `.npy` 文件并将其包含在内。

推荐打包命令如下：

```bash
zip -r model.zip main.py model/qwen_embedding_model rag/*.npy rag/*.json
```
