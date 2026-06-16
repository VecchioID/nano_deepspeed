import torch
import torch.nn as nn

class ManualZeRO1(nn.Module):
    def __init__(self, model, rank, world_size):
        super().__init__()
        self.model = model
        self.rank = rank
        self.world_size = world_size

        all_params = list(model.parameters())
        total_params = sum(p.numel() for p in all_params)
        params_per_rank = total_params // world_size
        self.start_idx = rank * params_per_rank
        self.end_idx = (rank + 1) * params_per_rank if rank < world_size - 1 else total_params
        self.n_local = self.end_idx - self.start_idx

        self.flat_params = torch.cat([p.data.reshape(-1).float() for p in all_params])
        self.local_params = self.flat_params[self.start_idx:self.end_idx].clone()
        self.momentum = torch.zeros(self.n_local)
        self.variance = torch.zeros(self.n_local)

    def step(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        flat_grads = torch.cat([
            p.grad.data.reshape(-1).float() for p in self.model.parameters()
        ])
        local_grads = flat_grads[self.start_idx:self.end_idx]

        self.momentum.mul_(beta1).add_(local_grads, alpha=1 - beta1)
        self.variance.mul_(beta2).add_(local_grads ** 2, alpha=1 - beta2)

        m_hat = self.momentum / (1 - beta1)
        v_hat = self.variance / (1 - beta2)
        update = m_hat / (v_hat.sqrt() + eps)
        self.local_params.add_(update, alpha=-lr)

        offset = 0
        for p in self.model.parameters():
            n = p.numel()
            local_start = max(0, self.start_idx - offset)
            local_end = min(n, self.end_idx - offset)
            if local_end > local_start:
                p.data.reshape(-1)[local_start:local_end] = \
                    self.local_params[local_start - (self.start_idx - offset):
                                      local_end - (self.start_idx - offset)].to(p.dtype)
            offset += n
        return local_grads.norm().item()


def demo():
    print("=" * 60)
    print("ZeRO-1: 优化器状态分片 Demo")
    print("=" * 60)
    world_size = 4
    model = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 10))

    optimizers = []
    for rank in range(world_size):
        opt = ManualZeRO1(model, rank, world_size)
        optimizers.append(opt)

    x = torch.randn(16, 128)
    y = torch.randint(0, 10, (16,))

    for rank in range(world_size):
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()
        grad_norm = optimizers[rank].step(lr=0.01)
        print(f"  [Rank {rank}] 本地梯度范数={grad_norm:.4f}, 本地参数大小={optimizers[rank].n_local}")

    print()
    print("关键观察:")
    print("  - 每卡 Adam 状态: momentum(4Ψ/N) + variance(4Ψ/N) = 8Ψ/N")
    print("  - 而 DDP: 每卡 8Ψ (重复 N 次)")
    print(f"  总节省 = 8Ψ × (N-1)/N = {8*(world_size-1)/world_size}Ψ")

if __name__ == "__main__":
    demo()
