import torch
import torch.nn as nn

class ShardedParameter:
    def __init__(self, data: torch.Tensor, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.full_shape = data.shape
        self.full_numel = data.numel()

        chunk_size = self.full_numel // world_size
        remainder = self.full_numel % world_size
        self.start = rank * chunk_size + min(rank, remainder)
        self.end = self.start + chunk_size + (1 if rank < remainder else 0)

        flat = data.reshape(-1)
        self.local_shard = flat[self.start:self.end].clone()
        self.local_numel = self.local_shard.numel()

    def all_gather(self) -> torch.Tensor:
        full = torch.zeros(self.full_numel)
        full[self.start:self.end] = self.local_shard
        return full.reshape(self.full_shape)

    def free_except_shard(self):
        pass

class ManualZeRO3Model(nn.Module):
    def __init__(self, rank: int, world_size: int, d_model=64, n_layers=2):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.layers = nn.ModuleList([
            nn.Linear(d_model, d_model * 4) if i % 2 == 0 else nn.Linear(d_model * 4, d_model)
            for i in range(n_layers * 2)
        ])
        self.sharded = {}
        for name, p in self.named_parameters():
            d = p.data
            self.sharded[name] = ShardedParameter(p.data, rank, world_size)

    def forward(self, x):
        for name, layer in self.named_children():
            n, s = self.sharded.get(name, None), None
            # simulate All-Gather
            for pn, ps in self.sharded.items():
                if pn.startswith(name):
                    _ = ps.all_gather()
            x = layer(x)
        return x

def demo():
    print("=" * 60)
    print("ZeRO-3: 参数分片 + 动态物化 Demo")
    print("=" * 60)
    world_size = 4

    models = [ManualZeRO3Model(r, world_size, d_model=64, n_layers=2) for r in range(world_size)]
    total = sum(p.numel() for p in models[0].parameters())

    print(f"\n模型总参数: {total:,}")
    print(f"每卡本地参数 (ZeRO-3): {total // world_size:,} ({total//world_size/total*100:.0f}%)")
    print()
    print("显存对比 (模拟 Ψ=65B, N=64):")
    print(f"  DDP 每卡:      16Ψ = {16*65:.0f} GB")
    print(f"  ZeRO-3 每卡:  16Ψ/N = {16*65/64:.1f} GB")
    print()
    print("通信量对比:")
    print(f"  DDP:    All-Reduce    2Ψ")
    print(f"  ZeRO-3: All-Gather×2 + Reduce-Scatter = 3Ψ\n")

if __name__ == "__main__":
    demo()
