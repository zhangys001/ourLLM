"""
tools/retriever.py — 手写 Torch 余弦相似度检索

核心功能：给定查询向量与知识库向量矩阵，计算余弦相似度并返回 Top-K 结果。
该模块最终将内联到 main.py 中，此处作为开发调试用。
"""

import torch


def cosine_similarity_topk(
    query_emb: torch.Tensor,
    knowledge_embs: torch.Tensor,
    top_k: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    计算查询向量与知识库向量的余弦相似度，返回 Top-K 索引和分数。

    Args:
        query_emb:     查询向量, shape (dim,) 或 (1, dim)
        knowledge_embs: 知识库向量矩阵, shape (N, dim)
        top_k:         返回的最大结果数

    Returns:
        topk_scores: Top-K 余弦相似度分数, shape (top_k,)
        topk_indices: Top-K 对应的知识库索引, shape (top_k,)
    """
    # ---- 1. 维度检查与对齐 ----
    if query_emb.dim() == 1:
        query_emb = query_emb.unsqueeze(0)  # (dim,) → (1, dim)

    if knowledge_embs.dim() == 1:
        knowledge_embs = knowledge_embs.unsqueeze(0)

    # ---- 2. L2 归一化 ----
    # 沿最后一维做 L2 归一化, 使 ||v||₂ = 1
    query_norm = torch.nn.functional.normalize(query_emb, p=2, dim=-1)       # (1, dim)
    knowledge_norm = torch.nn.functional.normalize(knowledge_embs, p=2, dim=-1)  # (N, dim)

    # ---- 3. 余弦相似度 = 归一化向量内积 ----
    # (1, dim) @ (dim, N) → (1, N) → squeeze → (N,)
    scores = (query_norm @ knowledge_norm.T).squeeze(0)

    # ---- 4. Top-K ----
    topk_scores, topk_indices = torch.topk(scores, k=min(top_k, scores.shape[0]))

    return topk_scores, topk_indices


# ============================================================
# 简单自测
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("余弦相似度 Top-K 检索 — 自测")
    print("=" * 50)

    # 构造模拟数据
    torch.manual_seed(42)
    dim = 128
    num_knowledge = 10
    top_k = 3

    # 随机知识库向量
    knowledge = torch.randn(num_knowledge, dim)

    # 取知识库第一条作为查询（略微加噪声）, 期望它排第一
    query = knowledge[0] + torch.randn(dim) * 0.01

    scores, indices = cosine_similarity_topk(query, knowledge, top_k=top_k)

    print(f"\n知识库向量数: {num_knowledge}, 向量维度: {dim}")
    print(f"查询向量 (前 5 维): {query[:5]}")
    print(f"\nTop-{top_k} 结果:")
    for rank, (score, idx) in enumerate(zip(scores, indices), start=1):
        print(f"  #{rank}  idx={idx.item():3d}  cos_sim={score.item():.6f}")

    # 验证: 最高分应该接近 1.0, 且索引为 0
    assert indices[0].item() == 0, f"期望 idx=0 排第一, 实际 idx={indices[0].item()}"
    assert scores[0].item() > 0.99, f"期望余弦相似度 > 0.99, 实际 {scores[0].item():.6f}"
    print("\n✅ 自测通过: 余弦相似度计算正确")
