"""
example_01_basic_deepspeed.py
DeepSpeed 基础集成示例

用法:
    deepspeed --num_gpus=2 example_01_basic_deepspeed.py \
        --deepspeed_config ds_config.json

演示目的:
    - ZeRO-2 基本使用
    - DeepSpeed Engine 初始化
    - 训练循环替换
"""

import argparse
import torch
import torch.nn as nn
import deepspeed


class TinyTransformer(nn.Module):
    """最小的 Transformer 用于演示"""

    def __init__(self, vocab_size=32000, hidden=768, num_layers=6, num_heads=12):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden,
                nhead=num_heads,
                dim_feedforward=hidden * 4,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
            )
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.embed.weight = self.head.weight  # weight tying

    def forward(self, input_ids):
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits


def get_data_loader(batch_size=4, seq_len=512, vocab_size=32000, num_batches=100):
    """生成模拟训练数据"""
    for _ in range(num_batches):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        labels = input_ids.clone()
        yield {"input_ids": input_ids.cuda(), "labels": labels.cuda()}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    parser = deepspeed.add_config_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. 创建模型
    model = TinyTransformer()

    # 2. 定义 DeepSpeed 配置 (也可以从文件加载)
    ds_config = {
        "train_batch_size": 32,
        "train_micro_batch_size_per_gpu": 4,
        "gradient_accumulation_steps": 8,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            },
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 3e-5,
                "warmup_num_steps": 100,
            },
        },
        "zero_optimization": {"stage": 2},
        "fp16": {"enabled": True},
        "gradient_clipping": 1.0,
    }

    # 3. 初始化 DeepSpeed Engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config_params=ds_config,
    )

    # 4. 训练循环
    print(f"Rank {args.local_rank}: Starting training...")
    model_engine.train()

    for step, batch in enumerate(get_data_loader()):
        input_ids = batch["input_ids"]
        labels = batch["labels"]

        # 前向
        logits = model_engine(input_ids)

        # 计算 loss
        loss = nn.CrossEntropyLoss()(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        # 反向 (DeepSpeed 封装)
        model_engine.backward(loss)

        # 参数更新
        model_engine.step()

        if step % 10 == 0 and args.local_rank == 0:
            print(f"Step {step}: loss = {loss.item():.4f}")

        if step >= 50:
            break

    # 5. 保存 checkpoint
    model_engine.save_checkpoint("./checkpoints", client_state={"step": step})
    print("Training complete!")


if __name__ == "__main__":
    main()
