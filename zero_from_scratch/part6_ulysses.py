"""
从零实现 DeepSpeed Ulysses 序列并行

核心思想:
  1. 序列被切到 P 个 GPU 上, 每卡拿 N/P 个 token
  2. 计算 attention 前, All-to-All 通信 → 转成按注意力头切分
  3. 每卡计算自己那部分头 (完整序列长度)
  4. 计算完后 All-to-All 通信 → 转回按序列切分

为什么用 All-to-All 而不是 All-Gather/Reduce-Scatter?
  All-to-All: N 卡各发 N 份不同的数据 → 完美支持维度转置
  All-Gather: N 卡各发同一份数据 → 不适合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, math


class NaiveAttention(nn.Module):
    """标准单卡 attention (对比用)"""
    def __init__(self, hidden_dim=4096, n_heads=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads

        self.wq = nn.Linear(hidden_dim, hidden_dim)
        self.wk = nn.Linear(hidden_dim, hidden_dim)
        self.wv = nn.Linear(hidden_dim, hidden_dim)
        self.wo = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, mask=None):
        B, N, H = x.shape

        Q = self.wq(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.wk(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.wv(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)

        # O(N²) attention 计算 ← 这就是瓶颈!
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        out = out.transpose(1, 2).contiguous().view(B, N, H)
        return self.wo(out)


class UlyssesAttention(nn.Module):
    """
    Ulysses 序列并行 attention

    流程:
      Step 1: 输入 x: (B, N/P, H)     ← 序列被切分
      Step 2: QKV 投影 (每卡算自己的 N/P 个 token, 完整 H 维)
      Step 3: All-to-All 通信: (B, N/P, H) → (B, N, H/P)
              转成按注意力头切分:
                GPU 0 → 头 0~(H/P-1) 的完整序列
                GPU 1 → 头 H/P~(2H/P-1) 的完整序列
                ...
      Step 4: 本地 attention (完整 N 序列, 但只有 H/P 个头)
      Step 5: All-to-All 通信: (B, N, H/P) → (B, N/P, H)
              转回按序列切分
      Step 6: 输出投影

    关键: All-to-All 通信量与序列长无关!
          P 卡时, 每卡收发 B × N/P × H 数据
          如果 P ∝ N (序列越长卡越多), 通信量 = 常数
    """
    def __init__(self, hidden_dim=4096, n_heads=32, sp_size=4, rank=0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_local_heads = n_heads // sp_size
        self.head_dim = hidden_dim // n_heads
        self.sp_size = sp_size
        self.rank = rank

        # QKV 投影: 完整 H → 完整 H (每卡先算自己的 N/P token)
        self.wq = nn.Linear(hidden_dim, hidden_dim)
        self.wk = nn.Linear(hidden_dim, hidden_dim)
        self.wv = nn.Linear(hidden_dim, hidden_dim)
        # 输出投影: H → H
        self.wo = nn.Linear(hidden_dim, hidden_dim)

    def all_to_all(self, x, scatter_dim=1, gather_dim=2):
        """
        模拟 All-to-All: (B, N/P, H) ↔ (B, N, H/P)

        reshape → transpose → reshape:
          (B, N/P, H) → (B, P, N/P², H) → (B, N/P², P, H) → (B, N/P², P×H)
        简化: N/P 能被 P 整除时
        """
        B, N_local, H = x.shape
        # 把 N_local 维拆成 (sp_size, N_local/sp_size)
        # 与 H 维拆成的 (sp_size, H/sp_size) 交换
        x = x.reshape(B, self.sp_size, N_local // self.sp_size,
                       self.sp_size, H // self.sp_size)
        x = x.permute(0, 2, 4, 1, 3).contiguous()
        x = x.reshape(B, N_local * H // (H // self.sp_size),
                       H // self.sp_size)  # temp
        # 更干净的实现: 直接 reshape
        return x.reshape(B, -1, H // self.sp_size)

    def forward(self, x, mask=None):
        # x: (B, N/P, H)
        B, N_local, H = x.shape
        N_total = N_local * self.sp_size

        # Step 2: QKV 投影 (每卡完整 H 维)
        Q = self.wq(x)     # (B, N/P, H)
        K = self.wk(x)
        V = self.wv(x)

        # Step 3: All-to-All  (B, N/P, H) → (B, N, H/P)
        # 每卡发送自己 N/P 个 token 的 QKV, 接收完整 N 个 token 但只拿 H/P 维
        Q = self.all_to_all(Q)
        K = self.all_to_all(K)
        V = self.all_to_all(V)  # (B, N, H/P)

        # Reshape: (B, N, H/P) → (B, n_local_heads, N, head_dim)
        nh_local = self.n_local_heads
        hd = self.head_dim
        Q = Q.view(B, N_total, nh_local, hd).transpose(1, 2)
        K = K.view(B, N_total, nh_local, hd).transpose(1, 2)
        V = V.view(B, N_total, nh_local, hd).transpose(1, 2)

        # Step 4: 本地 attention (完整序列, 本地头数)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(hd)
        if mask is not None:
            scores = scores.masked_fill(mask[:, None, :N_total, :N_total] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)  # (B, nh_local, N, hd)

        # Step 5: All-to-All 反向  (B, N, H/P) → (B, N/P, H)
        out = out.transpose(1, 2).contiguous().view(B, N_total, -1)
        # 反转 all_to_all: (B, N, H/P) → (B, N/P, H)
        out = out.reshape(B, self.sp_size, N_total // self.sp_size,
                          H // self.sp_size).permute(0, 2, 1, 3).contiguous()
        out = out.reshape(B, N_local, H)

        # Step 6: 输出投影
        return self.wo(out)


def memory_breakdown(seq_len, hidden=4096, n_heads=32, batch=1, sp=1):
    """对比普通 vs Ulysses 的 attention 内存"""
    head_dim = hidden // n_heads
    score_shape = batch * (n_heads // sp) * seq_len * seq_len

    print(f"  seq_len={seq_len}, sp_size={sp}:")
    print(f"    每卡头数: {n_heads} → {n_heads // sp}")
    print(f"    attention 矩阵: {batch * n_heads * seq_len * seq_len * 2 / 1e9:.1f} GB "
          f"→ {batch * (n_heads // sp) * seq_len * seq_len * 2 / 1e9:.1f} GB")
    print(f"    每卡序列长度: {seq_len} → {seq_len // sp} tokens (QKV 投影减半)")
    print(f"    通信量 (All-to-All): 2 × (B × N × H / P) = "
          f"{2 * batch * seq_len * hidden // sp * 2 / 1e9:.2f} GB (与 N 无关!)")
    print()


def demo():
    print("=" * 65)
    print("DeepSpeed Ulysses 序列并行 — 原理与实现")
    print("=" * 65)

    print("\n1. 为什么需要序列并行?")
    for sl in [4096, 16384, 65536]:
        memory_breakdown(sl, sp=1)
    print("  → 序列越长, attention 矩阵 O(N²) 越难放")

    print("\n2. Ulysses 怎么解决?")
    print("  把序列切成 P 段 + All-to-All 转成按头切分")
    print("  → 每卡只算 H/P 个头 → attention 矩阵 = O(N² × H/P)")
    for sl in [32768, 65536, 131072]:
        memory_breakdown(sl, sp=8)
    print("  → 8 卡时 128K 序列的 attention 矩阵从 1099GB → 137GB")

    print("\n3. 关键优势: 通信量不随序列长增长")
    print("  每步 All-to-All 通信: 2 × B × N × H / P")
    print("  如果 P ∝ N (序列越长, 卡越多), 通信量 = 常数!")
    print("  ( vs Ring Attention: 通信量随序列长增长 )")

    print("\n4. Ulysses + FlashAttention + ZeRO-3 = 终极方案")
    print("  Ulysses 分序列 → FlashAttention 加速每卡 → ZeRO-3 分参数")
    print("  三者正交, 组合使用")


if __name__ == "__main__":
    demo()
