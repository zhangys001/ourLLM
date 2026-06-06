# 智能计算系统大作业 — RAG 问答系统

两阶段 RAG 问答系统：向量相似检索（Qwen Embedding）→ LLM 增强生成（Qwen2.5-3B）。

## 架构

```
问题 → Qwen Embedding → 向量
  ├─ Stage1: QA 题库 (embedding_qa) → cosine sim > 0.8 → 返回预制答案
  └─ Stage2: 知识库 (embedding_knowledge) → top-5 > 0.5 → LLM 约束生成
       └─ 无命中 → LLM 降级为"自由发挥"模式（无知识约束）
```

- **Stage1** 处理原题命中（306 条 QA，来自公开习题），阈值精确匹配原题拿满时间分。
- **Stage2** 检索知识片段（1121 条，课程 slides/文档清洗），喂入 LLM prompt 作为约束参考。
- **LLM 懒加载** — 仅 Stage1 未命中时触发。

## 目录结构

```text
.
├── main.py                     # 评测入口，answer(question, debug) → str
├── model/
│   ├── qwen_embedding_model/   # Qwen Embedding 模型权重
│   └── qwen2.5-3B/             # Qwen2.5-3B 大语言模型权重
├── data/                       # 原始数据与清洗产物（不提交）
│   ├── clean_data/             # 清洗后的 markdown 与 QA 文件
│   ├── raw_md/                 # 原始 markdown 格式文档
│   └── raw_pdf/                # 原始 PDF 课件
├── rag/                        # 运行时向量数据（提交）
│   ├── embedding_knowledge.npy # 知识库向量矩阵（L2 归一化）
│   ├── embedding_knowledge.json# 知识库文本（1121 条）
│   ├── embedding_qa.npy        # QA 题库向量矩阵（L2 归一化）
│   └── embedding_qa.json       # QA 题库文本（306 条）
├── tools/                      # 辅助脚本（不提交）
│   └── make_npy.py             # 离线向量生成：读 JSON → encode → 存 .npy
├── plans/                      # 方案规划文档
└── CLAUDE.md                   # 项目开发指引
```

## 上手指南

1. **模型下载**：将 Qwen Embedding 模型放到 `model/qwen_embedding_model/`，Qwen2.5-3B 放到 `model/qwen2.5-3B/`。
2. **环境**：使用 `ourLLM_env`（micromamba），不依赖 pip/poetry/pipenv。
3. **生成向量**：首次运行前执行 `python tools/make_npy.py`（.npy 被 .gitignore 忽略，需本地生成）。
4. **测试**：
   ```bash
   python main.py "问题文本"           # 静默模式，结果输出到 stdout
   python main.py "问题文本" --debug   # 调试模式，相似度/命中详情输出到 stderr
   ```
   静默模式下：`TQDM_DISABLE=1`，transformers logging 设为 ERROR，stderr 重定向到 `/dev/null`——评测机将 stderr 视为错误，无害 warning 会扣分。

## 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `QA_THRESHOLD` | 0.8 | QA 题库命中阈值 |
| `KNOWLEDGE_THRESHOLD` | 0.5 | 知识库检索阈值 |
| `KNOWLEDGE_TOP_K` | 5 | 单次查询最大知识片段数 |

## 当前状态

项目基本完成。后续工作重点：
- 根据评测机反馈修正 `rag/` 知识库内容
- 微调 `_build_prompt()` 中的 prompt 模板

## 打包提交

提交内容仅需：`main.py` + `model/`（两个模型权重）+ `rag/*.npy` + `rag/*.json`。`data/`、`tools/`、`plans/` 不打包。

确保本地已生成 `.npy` 文件后：

```bash
7z a -tzip -mmt=12 -x!model/qwen_model model.zip main.py model/qwen_embedding_model/ model/qwen2.5-3B/ rag/*.npy rag/*.json
```

> `rag/*.npy` 显式列出——若忘记生成 `.npy`，通配符匹配为空，7z 直接报错，避免静默打包残缺的 rag 目录。
