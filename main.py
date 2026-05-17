"""
main.py — 嵌入检索入口

用法:
    python main.py "问题文本"              # 普通模式
    python main.py "问题文本" --debug       # 调试模式: 显示详细相似度 + 加载进度
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

# ── 调试模式: 在 import transformers 之前设环境变量 ──
DEBUG = "--debug" in sys.argv
if not DEBUG:
    import os
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

from transformers import AutoModel, AutoTokenizer

if not DEBUG:
    import transformers
    transformers.logging.set_verbosity_error()

# ── 路径 ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model" / "qwen_embedding_model"
RAG_DIR = BASE_DIR / "rag"


def _load_model(device: str | None = None):
    """加载嵌入模型, 返回 (model, tokenizer, device)"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype="auto"
    ).to(device)
    model.eval()
    return model, tokenizer, device


def _load_rag():
    """加载 RAG 向量 + JSON, 返回 (emb_matrix, items)"""
    npy_path = RAG_DIR / "embedding_qa.npy"
    json_path = RAG_DIR / "embedding_qa.json"

    if not npy_path.exists() or not json_path.exists():
        return None, None

    emb_matrix = np.load(str(npy_path)).astype(np.float32)
    with open(json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    return emb_matrix, items


@torch.no_grad()
def _encode(model, tokenizer, device, text: str) -> np.ndarray:
    """
    单条文本 → (1024,) float32 归一化向量.

    池化方式: last token (Qwen3-Embedding 配置).
    """
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(**inputs)
    emb = outputs.last_hidden_state[:, -1]  # last token
    emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    return emb.cpu().float().numpy().flatten()


def _cosine_similarity(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """余弦相似度, query (dim,), candidates (N, dim) → (N,)"""
    q_norm = query / np.linalg.norm(query)
    c_norm = candidates / np.linalg.norm(candidates, axis=1, keepdims=True)
    return c_norm @ q_norm


def answer(question: str, debug: bool = False) -> str:
    """
    核心入口:
      1. 加载嵌入模型
      2. 编码问题向量
      3. 与 RAG 库余弦相似度检索
      4. 最高相似度 > 0.8 则返回对应 Answer, 否则返回兜底

    debug=True: 输出详细相似度到 stderr.
    """
    # ── 加载模型 ──
    try:
        model, tokenizer, device = _load_model()
    except Exception as e:
        return f"模型加载失败: {e}"

    # ── 加载 RAG ──
    emb_matrix, items = _load_rag()
    if emb_matrix is None or items is None:
        return "没找到匹配项"

    # ── 编码问题 ──
    try:
        q_vec = _encode(model, tokenizer, device, question)
    except Exception as e:
        return f"编码失败: {e}"

    # ── 余弦相似度检索 ──
    scores = _cosine_similarity(q_vec, emb_matrix)

    # ── 调试输出 ──
    if debug:
        ranked = np.argsort(scores)[::-1]
        _print_debug(question, items, scores, ranked)

    # ── 找最优匹配 ──
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])

    if best_score > 0.8:
        return items[best_idx].get("Answer", "没找到匹配项")
    else:
        return "没找到匹配项"


def _print_debug(question, items, scores, ranked):
    """打印详细相似度到 stderr"""
    import sys as _sys
    out = _sys.stderr  # 不污染 stdout 的回答

    print("=" * 70, file=out)
    print(f"查询: {question}", file=out)
    print("=" * 70, file=out)
    print(f"{'排名':<6}{'相似度':<10}{'npyID':<8}{'Question (前60字)'}", file=out)
    print(f"{'-'*6:<6}{'-'*10:<10}{'-'*8:<8}{'-'*50}", file=out)

    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])
    for rank, idx in enumerate(ranked):
        item = items[idx]
        marker = " ← 命中" if rank == 0 and best_score > 0.8 else ""
        print(
            f"{rank + 1:<6}{float(scores[idx]):.4f}{'':>4}"
            f"{item.get('npyID', ''):<8}{item.get('Question', '')[:60]}{marker}",
            file=out,
        )

    print(file=out)
    print(f"最高相似度: {best_score:.4f}", file=out)
    print(f"阈值 0.8: {'✓ 超过' if best_score > 0.8 else '✗ 未超过'}", file=out)
    print("=" * 70, file=out)


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("用法: python main.py \"问题文本\" [--debug]", file=sys.stderr)
        sys.exit(1)

    # 去掉 --debug 参数后取真正的消息
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    question = args[0].strip() if args else ""

    result = answer(question, debug=DEBUG).strip()

    if not result:
        result = "没找到匹配项"

    print(result)


if __name__ == "__main__":
    main()
