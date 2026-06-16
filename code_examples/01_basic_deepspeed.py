"""
01_basic_deepspeed.py
DeepSpeed 基础使用示例: 训练一个简单的 Transformer 模型

运行:
  deepspeed --num_gpus=2 01_basic_deepspeed.py --deepspeed_config ds_config_basic.json
"""

import torch
import torch.nn as nn
import deepspeed
import argparse


class SimpleTransformer(nn.Module):
    """极简 Transformer 用于演示"""

    def __init__(self, vocab_size=10000, hidden_dim=1024, num_layers=4, num_heads=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)

        self.embedding.weight = self.lm_head.weight  # weight tying

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    # 模型: 约 210M 参数
    model = SimpleTransformer(
        vocab_size=10000,
        hidden_dim=1024,
        num_layers=4,
        num_heads=16,
    )

    # 假数据
    batch_size = 8
    seq_len = 512
    dummy_input = torch.randint(0, 10000, (batch_size, seq_len)).cuda()
    dummy_labels = torch.randint(0, 10000, (batch_size, seq_len)).cuda()

    # DeepSpeed 初始化
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config_params="ds_config_basic.json",
    )

    print(f"DeepSpeed ZeRO stage: {model_engine.zero_optimization_stage()}")
    print(f"World size: {model_engine.world_size}")
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    for step in range(100):
        outputs = model_engine(dummy_input)
        loss = nn.functional.cross_entropy(
            outputs.view(-1, outputs.size(-1)),
            dummy_labels.view(-1),
        )

        model_engine.backward(loss)
        model_engine.step()

        if step % 10 == 0 and model_engine.global_rank == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")


if __name__ == "__main__":
    train()
