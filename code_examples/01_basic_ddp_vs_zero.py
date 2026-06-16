#!/usr/bin/env python3
"""
01_basic_ddp.py - 基础 DDP 与 ZeRO 对比

演示: 标准 DDP vs ZeRO-2 vs ZeRO-3 的显存对比
使用方法:
    torchrun --nproc_per_node=2 01_basic_ddp.py --zero_stage 2
"""

import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset


class TinyTransformer(nn.Module):
    """微型 Transformer 用于演示"""
    def __init__(self, vocab_size=1000, d_model=256, n_layers=4, n_heads=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=d_model*4, batch_first=True)
            for _ in range(n_layers)
        ])
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


class RandomDataset(Dataset):
    def __init__(self, vocab_size, seq_len, size=1000):
        self.data = torch.randint(0, vocab_size, (size, seq_len))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.data[idx]


def train_with_deepspeed(args):
    import deepspeed

    # 创建模型
    model = TinyTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
    )

    # DeepSpeed 配置
    ds_config = {
        "train_batch_size": args.batch_size * args.grad_accum_steps,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accum_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 3e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01,
            }
        },
        "zero_optimization": {
            "stage": args.zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "fp16": {
            "enabled": args.fp16,
        },
    }

    # 初始化 DeepSpeed
    model_engine, optimizer, train_loader, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config_params=ds_config,
    )

    # 打印显存信息
    rank = dist.get_rank()
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"ZeRO Stage: {args.zero_stage}")
        print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Grad accumulation: {args.grad_accum_steps}")
        print(f"{'='*60}\n")

    # 训练循环
    for step, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(model_engine.device)
        labels = labels.to(model_engine.device)

        outputs = model_engine(inputs)
        loss = nn.CrossEntropyLoss()(outputs.view(-1, args.vocab_size), labels.view(-1))

        model_engine.backward(loss)
        model_engine.step()

        if step >= args.max_steps:
            break

    # 打印最终显存
    if rank == 0:
        mem_allocated = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nPeak GPU memory: {mem_allocated:.2f} GB")
        print(f"Model size: {args.d_model}, Layers: {args.n_layers}")


def train_with_torch_ddp(args):
    """标准 DDP 训练 (对比用)"""
    from torch.nn.parallel import DistributedDataParallel as DDP

    model = TinyTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
    ).cuda()

    model = DDP(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    dataset = RandomDataset(args.vocab_size, args.seq_len)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    for step, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs.view(-1, args.vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step >= args.max_steps:
            break


def train_with_fsdp(args):
    """FSDP 训练 (对比用)"""
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    from torch.distributed.fsdp import ShardingStrategy

    model = TinyTransformer(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
    )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        auto_wrap_policy=transformer_auto_wrap_policy,
        device_id=torch.cuda.current_device(),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    dataset = RandomDataset(args.vocab_size, args.seq_len)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler)

    for step, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.cuda(), labels.cuda()
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs.view(-1, args.vocab_size), labels.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step >= args.max_steps:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero_stage", type=int, default=2, choices=[0, 1, 2, 3])
    parser.add_argument("--method", type=str, default="deepspeed", choices=["deepspeed", "ddp", "fsdp"])
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=50)
    parser.add_argument("--vocab_size", type=int, default=1000)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    if args.method == "deepspeed":
        train_with_deepspeed(args)
    elif args.method == "ddp":
        dist.init_process_group(backend="nccl")
        train_with_torch_ddp(args)
    elif args.method == "fsdp":
        dist.init_process_group(backend="nccl")
        train_with_fsdp(args)
