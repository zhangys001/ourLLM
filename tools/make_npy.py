"""
tools/make_npy.py — 使用 Qwen3 嵌入模型生成 .npy 向量文件

用法:
    micromamba run -n ourLLM_env python tools/make_npy.py

流程:
    扫描 rag/ 目录下的 .json 文件 → 解析 → 对指定字段算向量 → 输出 .npy + 回填 npyID
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# ── 路径 ─────────────────────────────────────────────────────
PROJ_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJ_DIR / "model" / "qwen_embedding_model"
RAG_DIR = PROJ_DIR / "rag"


def load_embedding_model(device: str | None = None) -> tuple:
    """加载 Qwen3 嵌入模型, local_files_only=True"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        MODEL_DIR, local_files_only=True, torch_dtype="auto"
    ).to(device)
    model.eval()
    return model, tokenizer, device


@torch.no_grad()
def embed_texts(
    model, tokenizer, texts: list[str], device: str
) -> np.ndarray:
    """
    对文本列表批量编码, 返回 float32 的 (N, 1024) 向量矩阵.

    文档向量: 直接编码文本, 无前缀.
    取 last_hidden_state[:, -1] (last token) 后 L2 归一化.
    """
    inputs = tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
    ).to(device)

    outputs = model(**inputs)
    # (N, seq_len, 1024) → (N, 1024), 取最后 token (decoder-only 池化)
    embeddings = outputs.last_hidden_state[:, -1]
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
    return embeddings.cpu().float().numpy().astype(np.float32)


def build_embeddings(
    model, tokenizer, device: str,
    items: list[dict],
    field: str,
    output_npy: Path,
    json_path: Path,
) -> None:
    """
    核心函数: 对 items 列表的 field 字段计算 embedding.

    Args:
        items:   JSON 解析后的列表, 每个元素是 dict
        field:   要编码的字段名, 如 "Question"
        output_npy: 输出 .npy 文件路径
        json_path:  回填后写回的原 JSON 文件路径
    """
    # ── 提取文本 ──
    texts = [item[field] for item in items]
    print(f"  待编码条目: {len(texts)}, 字段: '{field}'")

    if not texts:
        print("  ⚠  空列表, 跳过")
        return

    # ── 编码 ──
    emb_matrix = embed_texts(model, tokenizer, texts, device)
    np.save(str(output_npy), emb_matrix)
    print(f"  向量矩阵: {emb_matrix.shape} → {output_npy}")

    # ── 回填 npyID ──
    for i, item in enumerate(items):
        item["npyID"] = str(i)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=4)
    print(f"  npyID 已回填 → {json_path}")


def process_single_json(model, tokenizer, device: str, json_filename: str, field: str) -> None:
    """
    处理单个 JSON 文件, 对指定字段编码并回填 npyID.

    Args:
        json_filename: JSON 文件名 (如 "embedding_qa.json"), 位于 rag/ 目录下
        field:         要编码的字段名 (如 "Question", "content")
    """
    jp = RAG_DIR / json_filename
    if not jp.exists():
        print(f"⚠  未找到文件: {jp}")
        return

    print(f"\n{'='*60}")
    print(f"处理: {jp.name} (目标字段: {field})")
    print(f"{'='*60}")

    with open(jp, "r", encoding="utf-8") as f:
        items = json.load(f)

    if not isinstance(items, list):
        print(f"  ⚠  跳过: 顶层不是 JSON 数组")
        return

    output_npy = jp.with_suffix(".npy")

    if not items:
        # 空列表: 删旧 .npy 避免 stale 数据
        if output_npy.exists():
            output_npy.unlink()
            print(f"  🗑  空列表, 删除旧 .npy: {output_npy.name}")
        else:
            print(f"  ⚠  空列表, 跳过")
        return

    build_embeddings(model, tokenizer, device, items, field, output_npy, jp)


# ── 也可对单条文本编码, 供 tools/ 调试用 ──
@torch.no_grad()
def encode_single(model, tokenizer, device, text: str) -> np.ndarray:
    """编码单条文本, 返回 (1024,) float32 向量"""
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(**inputs)
    emb = outputs.last_hidden_state[:, -1]  # last token
    emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
    return emb.cpu().numpy().flatten()


def main():
    print("Qwen3 嵌入模型 — 批量编码工具")
    print(f"模型: {MODEL_DIR}")

    model, tokenizer, device = None, None, None

    while True:
        print("\n" + "="*40)
        print("请选择要执行的操作:")
        print("1. 制作所有npy文件")
        print("2. 制作QA的npy文件")
        print("3. 制作主知识库的npy文件")
        print("4. 制作其他npy")
        print("0. 退出")
        print("="*40)
        
        choice = input("请输入数字选项 (0-4): ").strip()
        
        if choice in ['1', '2', '3', '4'] and model is None:
            model, tokenizer, device = load_embedding_model()
            print(f"设备: {device}, 参数量: {sum(p.numel() for p in model.parameters()):,}")

        # 使用 str() 包裹 device 解决 Pylance 类型推断问题
        # 虽然经过 load_embedding_model 后必定不为 None, 但初始赋值让其认为是 Unknown | None
        if choice == '1':
            process_single_json(model, tokenizer, str(device), "embedding_qa.json", "Question")
            process_single_json(model, tokenizer, str(device), "embedding_knowledge.json", "content")
            print("\n✅ 所有指定文件处理完成")
        elif choice == '2':
            process_single_json(model, tokenizer, str(device), "embedding_qa.json", "Question")
            print("\n✅ QA文件处理完成")
        elif choice == '3':
            process_single_json(model, tokenizer, str(device), "embedding_knowledge.json", "content")
            print("\n✅ 主知识库文件处理完成")
        elif choice == '4':
            filename = input("请输入 rag/ 目录下的 .json 文件名 (例如 abc.json): ").strip()
            field = input("请输入需要向量化的字段名 (例如 content): ").strip()
            if filename and field:
                process_single_json(model, tokenizer, str(device), filename, field)
                print(f"\n✅ {filename} 处理完成")
            else:
                print("⚠  文件名或字段名不能为空")
        elif choice == '0':
            print("退出程序。")
            break
        else:
            print("⚠  无效选项，请重新输入")


if __name__ == "__main__":
    main()
