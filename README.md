# 智能计算系统大作业 — Qwen2 + RAG

## 目录结构

```
.
├── main.py                 # 评测入口
├── model/
│   ├── qwen_model/         # Qwen2-0.5B 权重
│   └── bge_model/          # BGE-small-zh-v1.5 权重
├── data/                   # 原始知识 JSON (不提交)
│   └── knowledge.json
├── rag/                    # 运行时向量数据 (提交)
│   ├── embeddings.npy
│   └── embedding.json
├── tools/                  # 制作/辅助脚本 (不提交)
│   ├── build_embeddings.py
│   └── retriever.py
├── 方案规划.md              # 方案说明
└── 智能计算系统大作业任务书.md
```

## 上手指南

1. **模型下载**: 把 Qwen2-0.5B 放到 `model/qwen_model/`, BGE-small-zh-v1.5 放到 `model/bge_model/`
2. **环境**: 用 `ourLLM_env` (micromamba) 开发测试
3. **开发**: 按 `方案规划.md` 分阶段推进, 有问题群里问

## 打包

最终提交仅包含 `main.py` + `model/` + `rag/`。`data/` 和 `tools/` 不打包。

```bash
zip -r model.zip main.py model rag
```
