"""
example_03_custom_zero3.py
自定义 ZeRO-3 适配示例

演示目的:
    - ZeRO-3 下参数收集 (GatheredParameters)
    - Weight Tying 处理
    - 自定义 checkpointing
    - 分布式推理
"""

import torch
import torch.nn as nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import deepspeed
from deepspeed.zero import GatheredParameters
from deepspeed.runtime.activation_checkpointing import checkpointing


class TransformerBlock(nn.Module):
    """单个 Transformer 块"""

    def __init__(self, hidden=4096, num_heads=32):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            hidden, num_heads, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x):
        # 自注意力 + 残差
        attn_out, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x))
        x = x + attn_out
        # FFN + 残差
        x = x + self.ffn(self.norm2(x))
        return x


class ZeRO3CompatibleModel(nn.Module):
    """
    兼容 ZeRO-3 的自定义模型
    演示: weight tying + GatheredParameters
    """

    def __init__(self, vocab_size=32000, hidden=4096, num_layers=6):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList([
            TransformerBlock(hidden) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden)
        # 输出层与 embedding 共享权重 (weight tying)
        self.lm_head = nn.Linear(hidden, vocab_size, bias=False)
        self.lm_head.weight = self.embed_tokens.weight  # tied!

    def forward(self, input_ids):
        # 在 ZeRO-3 下, tied weights 需要特殊处理:
        # embedding 参数可能在另一个 rank 上,
        # 需要手动收集才能保证 tied weights 一致性
        with GatheredParameters(
            [self.embed_tokens.weight],
            modifier_rank=0,  # 只在 rank 0 上修改
        ):
            x = self.embed_tokens(input_ids)

        for layer in self.layers:
            # 对每个 Transformer 块启用 activation checkpointing
            x = checkpointing.checkpoint(layer, x)

        x = self.norm(x)

        with GatheredParameters([self.lm_head.weight]):
            logits = self.lm_head(x)

        return logits


def test_param_distribution(model):
    """验证 ZeRO-3 参数分片状态"""
    for name, param in model.named_parameters():
        if param._is_sharded:
            print(f"{name}: sharded, shape={param.shape}, "
                  f"local_shape={param.data.shape}")
        else:
            print(f"{name}: replicated, shape={param.shape}")


def test_tied_weights():
    """测试 tied weights 在 ZeRO-3 下的一致性"""
    model = ZeRO3CompatibleModel()
    assert model.embed_tokens.weight.data_ptr() == model.lm_head.weight.data_ptr()
    print("Weight tying: correct (same memory address)")


def main():
    # 测试模型定义
    model = ZeRO3CompatibleModel()
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    test_tied_weights()

    # 模拟单卡前向 (无 ZeRO, 用于对比)
    dummy_input = torch.randint(0, 32000, (2, 128))
    with torch.no_grad():
        output = model(dummy_input)
    print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()
