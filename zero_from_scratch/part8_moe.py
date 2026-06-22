"""
从零实现 DeepSpeed MoE (Mixture of Experts)

核心思想:
  标准 Dense 模型: 每个 token 激活全部参数
  MoE 模型:       每个 token 只激活 k 个 expert (k=1或2)

  效果: 参数量可以大很多, 但计算量几乎不变!
    例: Mixtral 8×7B = 47B 参数, 但每次只激活 2 个 expert = ~13B 计算
    参数量是 7B 的 7 倍, 计算量只多 1 倍
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Router(nn.Module):
    """
    门控网络 (Router/Gate)

    决定每个 token 发给哪几个 expert
    最简单的实现: 一个 Linear + Top-K + Softmax

    输入: (B, N, H)  →  输出: (B, N, num_experts)
    每行 = 该 token 选择每个 expert 的权重
    """
    def __init__(self, hidden_dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)

    def forward(self, x):
        # x: (B, N, H)  或  (total_tokens, H)
        scores = self.gate(x)          # (B, N, E) 或 (T, E)

        # Top-K: 选出分数最高的 k 个 expert
        top_k_weights, top_k_indices = torch.topk(
            F.softmax(scores, dim=-1), k=self.top_k, dim=-1
        )
        # top_k_weights: (B, N, k)  ← 每个 token 选中的 expert 权重
        # top_k_indices: (B, N, k)  ← 选中的 expert 编号

        return top_k_weights, top_k_indices


class Expert(nn.Module):
    """
    单个 Expert (通常是一个 FFN)

    MoE 中的 expert 就是标准 Transformer 的 FFN:
      FFN(x) = SwiGLU(W1(x)) · W2(x)  或  ReLU(W1(x)) · W2
    简单版: Linear + ReLU + Linear
    """
    def __init__(self, hidden_dim, ffn_dim):
        super().__init__()
        self.w1 = nn.Linear(hidden_dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, hidden_dim)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w2(self.activation(self.w1(x)))


class MoELayer(nn.Module):
    """
    MoE 层: Router + 多个 Expert

    流程:
      1. Router 决定每个 token 发给哪 k 个 expert
      2. token 按 expert 分组 (dispatch)
      3. 每个 expert 处理自己分到的 token
      4. token 按原顺序重组 (combine)
      5. 加权合并各 expert 输出
    """
    def __init__(self, hidden_dim, ffn_dim, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = Router(hidden_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            Expert(hidden_dim, ffn_dim) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: (B, N, H)
        B, N, H = x.shape
        x_flat = x.reshape(-1, H)   # (B*N, H) = (T, H)
        T = B * N

        # 1. Router: 每个 token 选 Top-2 expert
        weights, indices = self.router(x_flat)
        # weights: (T, k), indices: (T, k)

        # 2. Dispatch: token 按 expert 分组
        #    对每个 expert, 收集发給它的 token
        outputs = torch.zeros_like(x_flat)

        for expert_idx in range(self.num_experts):
            # 找到选了这个 expert 的 token
            mask = (indices == expert_idx).any(dim=-1)  # (T,)
            if mask.sum() == 0:
                continue

            # 拿到对应 token 和权重
            selected_tokens = x_flat[mask]              # (T_selected, H)

            # 找到该 expert 对应的权重
            # 每个 token 可能有多个 expert, 取出对当前 expert 的权重
            pos = (indices[mask] == expert_idx).int().argmax(dim=-1)
            expert_weight = weights[mask, pos].unsqueeze(-1)  # (T_selected, 1)

            # 3. Expert 计算
            expert_output = self.experts[expert_idx](selected_tokens)

            # 4. Combine: 加权合并
            outputs[mask] += expert_weight * expert_output

        return outputs.reshape(B, N, H)


# ====================================================================
# 完整 MoE Transformer 示例
# ====================================================================

class MoETransformerBlock(nn.Module):
    """带 MoE 的 Transformer 块"""
    def __init__(self, hidden_dim, num_heads, ffn_dim, num_experts, top_k=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # MoE 替换标准 FFN
        self.moe = MoELayer(hidden_dim, ffn_dim, num_experts, top_k)

    def forward(self, x):
        x = x + self.attention(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.moe(self.norm2(x))
        return x


class MoETransformer(nn.Module):
    """完整的 MoE Transformer 模型"""
    def __init__(self, vocab_size=10000, hidden_dim=512, num_heads=8,
                 num_layers=4, ffn_dim=2048, num_experts=8, top_k=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.blocks = nn.ModuleList([
            MoETransformerBlock(hidden_dim, num_heads, ffn_dim, num_experts, top_k)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x)


# ====================================================================
# Expert Parallelism (专家并行)
# ====================================================================

class ExpertParallelMoE:
    """
    模拟多卡 Expert Parallelism

    思路: 不同 GPU 放不同 expert
      标准: GPU 0-7 各放 1 个 expert (8 expert, 8 GPU)
      路由: token 被发到对应 expert 所在的 GPU

    通信: All-to-All (token 在 GPU 间转发)
    """
    def __init__(self, num_experts, num_gpus):
        self.num_experts = num_experts
        self.num_gpus = num_gpus
        # 每个 GPU 分到的 expert
        experts_per_gpu = num_experts // num_gpus
        self.gpu_to_experts = {
            gpu: list(range(gpu * experts_per_gpu, (gpu + 1) * experts_per_gpu))
            for gpu in range(num_gpus)
        }
        print("Expert Parallelism 分配:")
        for gpu, experts in self.gpu_to_experts.items():
            print(f"  GPU {gpu}: Expert {experts}")
        print()

    def dispatch(self, token_assignments):
        """
        token_assignments: {token_id: expert_id}
        返回每个 GPU 应该发送给其他 GPU 的 token 列表
        """
        # 构建 GPU → token 映射
        gpu_to_tokens = {gpu: [] for gpu in range(self.num_gpus)}
        for token_id, expert_id in token_assignments.items():
            for gpu, experts in self.gpu_to_experts.items():
                if expert_id in experts:
                    gpu_to_tokens[gpu].append(token_id)
                    break

        return gpu_to_tokens


# ====================================================================
# MoE vs Dense 对比
# ====================================================================

def compare_dense_vs_moe():
    """对比 Dense 模型和 MoE 模型的计算量和参数量"""
    H = 4096       # hidden dim
    ffn = 14336    # FFN hidden (LLaMA 比例 ~3.5×)
    E = 8          # expert 数
    k = 2          # top-k

    # Dense FFN: 1 个 FFN
    dense_params = 2 * (H * ffn + ffn + ffn * H + H)  # w1+b1 + w2+b2 简化

    # MoE FFN: E 个 expert, 每次激活 k 个
    moe_params = E * dense_params  # 参数量 × E
    moe_flops = k * dense_params    # 计算量 × k

    print(f"{'指标':<20} {'Dense':<15} {'MoE (8×2)':<15} {'比例':<10}")
    print("-" * 60)
    print(f"{'参数量':<20} {dense_params/1e6:<15.0f}M {moe_params/1e6:<15.0f}M {moe_params/dense_params:<10.1f}×")
    print(f"{'每次前向计算':<20} {dense_params/1e6:<15.0f}M {moe_flops/1e6:<15.0f}M {moe_flops/dense_params:<10.1f}×")
    print(f"{'实际效果':<20} {'基线':<15} {'参数量×8, 计算量×2':<15}")
    print()
    print("→ MoE 用 8 倍的参数量, 只付出 2 倍的计算代价")
    print("→ 这就是为什么 Mixtral 8×7B ≈ 47B 参数, 但推理成本接近 13B 模型")
    print()


def demo_moe_inference():
    """演示 MoE 前向传播, 显示路由选择"""
    print("=" * 60)
    print("MoE 前向传播: 观察 Router 的 Top-K 选择")
    print("=" * 60)

    B, N, H = 1, 4, 64   # 4 个 token
    num_experts = 4
    top_k = 2

    # 模拟输入
    x = torch.randn(B, N, H)

    router = Router(H, num_experts, top_k)
    weights, indices = router(x)

    print(f"\n输入: {B} batch × {N} token, hidden={H}")
    print(f"Experts: {num_experts}, Top-K: {top_k}")
    print()

    for token in range(N):
        print(f"  Token {token}:")
        print(f"    选中的 experts: {indices[0, token].tolist()}")
        print(f"    对应的权重:    {[f'{w:.3f}' for w in weights[0, token].tolist()]}")

    print()
    print("Key insight: 每个 token 只走 2/4 的 expert")
    print("另外 2 个 expert 完全不动 → 稀疏激活!")
    print()


def demo_expert_parallel():
    """模拟多卡 Expert Parallelism"""
    print("=" * 60)
    print("Expert Parallelism: 不同 GPU 放不同 Expert")
    print("=" * 60)

    ep = ExpertParallelMoE(num_experts=8, num_gpus=4)

    # 模拟 8 个 token 的路由决策
    assignments = {i: i % 8 for i in range(8)}  # token i → expert i % 8
    print(f"Token 路由决策: {assignments}")
    print()

    gpu_map = ep.dispatch(assignments)
    for gpu, tokens in gpu_map.items():
        print(f"  GPU {gpu} (Expert {ep.gpu_to_experts[gpu]}): "
              f"分配到 token {tokens}")
    print()
    print("通信: All-to-All 在各 GPU 间转发 token")
    print()


if __name__ == "__main__":
    compare_dense_vs_moe()
    demo_moe_inference()
    demo_expert_parallel()

    print("=" * 60)
    print("MoE 总结")
    print("=" * 60)
    print()
    print("1. 为什么用 MoE?")
    print("   在相同计算预算下, 使用更多参数 → 更好的效果")
    print()
    print("2. 核心组件")
    print("   - Router: 决定每个 token 去哪")
    print("   - Experts: 实际计算的 FFN")
    print("   - Top-K: 控制稀疏度 (k=2 最常用)")
    print()
    print("3. DeepSpeed-MoE 的优化")
    print("   - Expert Parallelism (EP): expert 分布到不同 GPU")
    print("   - E+D 并行: Expert + Data 并行组合")
    print("   - PR-MoE: 金字塔结构, 不同层不同 expert 数")
    print("   - ZeRO 兼容: expert 的参数继续用 ZeRO 分片")
    print()
    print("4. 谁在用?")
    print("   - Mixtral 8×7B (Mistral)")
    print("   - DeepSeekMoE (DeepSeek)")
    print("   - Switch Transformer (Google)")
    print("   - Qwen1.5-MoE (Alibaba)")
