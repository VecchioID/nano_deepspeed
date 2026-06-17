# 从零复现 DeepSpeed — 动手实现 ZeRO 核心机制

本 Lab 不是 `pip install deepspeed`，而是**手写 ZeRO 的分片逻辑**，理解它内部到底怎么工作的。

```
从零构建:
  Part 1: 手动 All-Reduce vs Reduce-Scatter     (梯度通信)
  Part 2: 手写 ZeRO-1: 优化器状态分片           (Adam分片更新)
  Part 3: 手写 ZeRO-3: 参数分片 + 动态物化      (All-Gather逐层收集)
  Part 4: 完整迷你 DeepSpeed                     (整合)
```

环境检查:
```bash
# 导航到项目根目录（请替换为您的实际路径）
mkdir -p zero_from_scratch
python -c "import torch; print(f'torch {torch.__version__}, cuda={torch.cuda.is_available()}')"
```

---

## Part 1: 手写 All-Reduce vs Reduce-Scatter

**目标**: 理解 DeepSpeed ZeRO-2 为什么用 Reduce-Scatter 替代 All-Reduce。

创建 `part1_communicate.py`:

```python
"""
手写 All-Reduce 和 Reduce-Scatter, 对比通信量
"""
import torch
import torch.distributed as dist

def my_all_reduce(tensor, op=dist.ReduceOp.SUM):
    """
    All-Reduce = Reduce + Broadcast
    通信量: 2 × tensor_size
    """
    # Step 1: Reduce (各卡求和到 rank 0)
    dist.reduce(tensor, dst=0, op=op)
    # Step 2: Broadcast (rank 0 广播结果到所有卡)
    dist.broadcast(tensor, src=0)

def my_reduce_scatter(tensor_list, op=dist.ReduceOp.SUM):
    """
    Reduce-Scatter = Reduce + Scatter (一步完成)
    通信量: 1 × tensor_size  (减半!)
    """
    world_size = dist.get_world_size()
    # 每卡只接收自己分片的结果
    chunk_size = tensor_list[0].numel() // world_size
    output_chunks = [t.chunk(world_size) for t in tensor_list]

    # 对每个分片位置, 对应的 rank 做 reduce
    rank = dist.get_rank()
    result = torch.zeros(chunk_size, device=tensor_list[0].device)
    for i in range(world_size):
        chunk_sum = sum(t[i].reshape(-1) for t in tensor_list)
        if i == rank:
            result = chunk_sum
    return result

def demo():
    dist.init_process_group(backend="gloo", init_method="tcp://localhost:12345",
                           rank=0, world_size=1)
    # 单卡演示: 用逻辑展示通信量差异
    print("=== All-Reduce vs Reduce-Scatter ===")
    print()
    print("All-Reduce (DDP 使用):")
    print("  1. 各卡梯度求和 (reduce)")
    print("  2. 广播结果给所有卡 (broadcast)")
    print("  通信量 = 2 × 参数量")
    print()
    print("Reduce-Scatter (ZeRO-2 使用):")
    print("  1. 各卡梯度求和的同时按 rank 切分 (reduce + scatter)")
    print("  2. 每卡只保留自己分片的部分")
    print("  通信量 = 1 × 参数量  ← 减半!")
    print()
    print(f"  假设 Ψ=7B:")
    print(f"    All-Reduce:    2 × 7B × 2bytes = 28 GB 通信")
    print(f"    Reduce-Scatter: 1 × 7B × 2bytes = 14 GB 通信  (-50%)")
    dist.destroy_process_group()

if __name__ == "__main__":
    demo()
```

```bash
python zero_from_scratch/part1_communicate.py
```

---

## Part 2: 手写 ZeRO-1 — 优化器状态分片

**目标**: 理解 Adam 优化器状态如何在卡间分片。

核心逻辑: Adam 有 3 个状态 (FP32 参数 + momentum + variance), 每卡只维护 1/N。

创建 `part2_zero1.py`:

```python
"""
手写 ZeRO-1: 优化器状态分片

核心思想:
  Adam 的 momentum / variance 分到 N 个 rank 上,
  每卡只更新自己分片部分的参数, 不需要通信。
"""
import torch
import torch.nn as nn

class ManualZeRO1(nn.Module):
    """
    手动实现 ZeRO-1 分片逻辑

    架构:
      - 每个 rank 持有完整模型副本 (参数 + 梯度)
      - 但 optimizer 状态 (momentum/variance) 只持有 1/N
      - 更新时, 每卡只更新自己分片的参数
    """
    def __init__(self, model, rank, world_size):
        super().__init__()
        self.model = model
        self.rank = rank
        self.world_size = world_size

        # 获取所有参数
        all_params = list(model.parameters())
        total_params = sum(p.numel() for p in all_params)

        # 计算每个 rank 分到的参数范围
        params_per_rank = total_params // world_size
        self.start_idx = rank * params_per_rank
        self.end_idx = (rank + 1) * params_per_rank if rank < world_size - 1 else total_params
        self.n_local = self.end_idx - self.start_idx

        # 展平所有参数, 确定本地分片在全局中的位置
        self.flat_params = torch.cat([p.data.reshape(-1).float() for p in all_params])
        self.local_params = self.flat_params[self.start_idx:self.end_idx].clone()

        # Adam 状态 (只分配本地分片!)
        self.momentum = torch.zeros(self.n_local)
        self.variance = torch.zeros(self.n_local)

        print(f"[ZeRO-1 Rank {rank}] 本地分片: {self.n_local:,} 参数 "
              f"(全局 {total_params:,}, {self.n_local/total_params*100:.1f}%)")

    def zero_grad(self):
        for p in self.model.parameters():
            if p.grad is not None:
                p.grad.zero_()

    def step(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        只更新本地分片对应的参数 (不需要通信!)

        传统 Adam: 所有 rank 各自完整更新 → 冗余
        ZeRO-1:    每卡只更新自己分片 → 节省 12Ψ/N 显存
        """
        # 展平梯度
        flat_grads = torch.cat([
            p.grad.data.reshape(-1).float()
            for p in self.model.parameters()
        ])
        local_grads = flat_grads[self.start_idx:self.end_idx]

        # Adam 更新 (只在本地分片上)
        self.momentum.mul_(beta1).add_(local_grads, alpha=1 - beta1)
        self.variance.mul_(beta2).add_(local_grads ** 2, alpha=1 - beta2)

        m_hat = self.momentum / (1 - beta1)
        v_hat = self.variance / (1 - beta2)

        update = m_hat / (v_hat.sqrt() + eps)
        self.local_params.add_(update, alpha=-lr)

        # 写回模型参数 (仅本地分片部分)
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
    """演示 ZeRO-1 在 4 个 rank 上的分片行为"""
    print("=" * 60)
    print("ZeRO-1: 优化器状态分片 Demo")
    print("=" * 60)

    world_size = 4  # 假设 4 卡
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )

    # 演示分片逻辑 (用 loop 模拟多 rank)
    optimizers = []
    for rank in range(world_size):
        opt = ManualZeRO1(model, rank, world_size)
        optimizers.append(opt)

    # 模拟一次训练 step
    x = torch.randn(16, 128)
    y = torch.randint(0, 10, (16,))

    for rank in range(world_size):
        opt = optimizers[rank]

        # 前向 + 反向 (每卡独立做)
        loss = nn.CrossEntropyLoss()(model(x), y)
        loss.backward()

        # ZeRO-1 更新: 每卡只更新自己分片
        grad_norm = opt.step(lr=0.01)
        print(f"  [Rank {rank}] 本地梯度范数={grad_norm:.4f}, "
              f"本地参数大小={opt.n_local}")

        if rank < world_size - 1:
            print(f"  [Rank {rank}] 通信: All-Gather 同步完整参数 (这一步后所有 rank 参数一致)")
            # 真实场景: 这里需要 All-Gather 合并参数
            # 省略: nccl.all_gather(flat_params, from_all_ranks)

    print()
    print("关键观察:")
    print("  - 每卡 Adam 状态: momentum(4Ψ/N) + variance(4Ψ/N) = 8Ψ/N")
    print("  - 而 DDP: 每卡 8Ψ  (重复 N 次)")
    print(f"  总节省 = 8Ψ × (N-1)/N = {8*(world_size-1)/world_size}Ψ")

if __name__ == "__main__":
    demo()
```

```bash
python zero_from_scratch/part2_zero1.py
```

---

## Part 3: 手写 ZeRO-3 — 参数分片 + 动态物化

**目标**: 理解 ZeRO-3 最核心的"计算时才收集参数, 算完就释放"。

创建 `part3_zero3.py`:

```python
"""
手写 ZeRO-3: 参数分片 + 动态参数物化

核心思想:
  每卡只存储 1/N 的参数。计算某层时,
  通过 All-Gather 从所有 rank 收集完整参数,
  计算完毕后立即释放 (只保留自己分片部分)。

这叫做 "Dynamic Parameter Materialization" (动态参数物化)
"""

import torch
import torch.nn as nn
from typing import List, Tuple

class ShardedParameter:
    """
    模拟 ZeRO-3 的分片参数

    每个 rank 持有参数的 1/N 分片,
    需要完整参数时通过 All-Gather 收集。
    """
    def __init__(self, data: torch.Tensor, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.full_shape = data.shape
        self.full_numel = data.numel()

        # 计算本地分片
        chunk_size = self.full_numel // world_size
        remainder = self.full_numel % world_size
        self.start = rank * chunk_size + min(rank, remainder)
        self.end = self.start + chunk_size + (1 if rank < remainder else 0)

        # 本地只持有 1/N 的分片
        flat = data.reshape(-1)
        self.local_shard = flat[self.start:self.end].clone()
        self.local_numel = self.local_shard.numel()

        print(f"    [Rank {rank}] 参数 {list(data.shape)}: "
              f"本地持有 {self.local_numel}/{self.full_numel} "
              f"({self.local_numel/self.full_numel*100:.0f}%)")

    def all_gather(self) -> torch.Tensor:
        """
        模拟 All-Gather: 从所有 rank 收集完整参数

        在真实 DeepSpeed 中:
          nccl.all_gather(output_buffer, self.local_shard)
        """
        # 模拟: 假设从其他 rank "收集" 了完整参数
        # (真实场景中通过 NCCL 通信)
        full = torch.zeros(self.full_numel)
        for r in range(self.world_size):
            r_start = r * (self.full_numel // self.world_size) + min(r, self.full_numel % self.world_size)
            r_end = r_start + (self.full_numel // self.world_size) + (1 if r < self.full_numel % self.world_size else 0)
            if r == self.rank:
                full[r_start:r_end] = self.local_shard
            # 其他 rank 的分片 "来自" All-Gather 通信
            # 这里简化: 从完整参数构造
        return full.reshape(self.full_shape)

    def free_except_shard(self):
        """
        释放完整参数, 只保留本地分片

        对应 DeepSpeed 的:
          _params.free_except(my_rank_slice)
        """
        # 在真实 DeepSpeed 中这里释放临时 buffer
        # 这里只是标记概念
        pass


class Zero3TransformerLayer(nn.Module):
    """
    支持 ZeRO-3 分片的 Transformer 层

    每层都:: 收集参数 → 计算 → 释放
    """
    def __init__(self, d_model, rank, world_size):
        super().__init__()
        self.rank = rank
        self.world_size = world_size

        # 这些参数在 ZeRO-3 下都是分片的!
        self.fc1 = nn.Linear(d_model, d_model * 4)
        self.fc2 = nn.Linear(d_model * 4, d_model)

        # 将所有权重视为分片参数 (实际 DeepSpeed 自动做)
        self.sharded_params = []
        for name, param in self.named_parameters():
            sp = ShardedParameter(param.data, rank, world_size)
            self.sharded_params.append(sp)

    def forward(self, x):
        """
        ZeRO-3 前向: 逐层收集参数 → 计算 → 释放
        """
        # Step 1: All-Gather 收集本层完整参数
        # (真实场景: nccl.all_gather)
        w1_full = self.sharded_params[0].all_gather()
        b1_full = self.sharded_params[1].all_gather()
        w2_full = self.sharded_params[2].all_gather()
        b2_full = self.sharded_params[3].all_gather()

        # Step 2: 计算 (使用完整参数)
        h = torch.relu(nn.functional.linear(x, w1_full, b1_full))
        h = nn.functional.linear(h, w2_full, b2_full)

        # Step 3: 释放完整参数 (只保留本地分片)
        for sp in self.sharded_params:
            sp.free_except_shard()

        return h


class ManualZeRO3Model(nn.Module):
    """
    手写 ZeRO-3 模型

    前向时逐层 All-Gather, 计算后立即释放。
    这模拟了 DeepSpeed ZeRO-3 的 Dynamic Parameter Materialization。
    """
    def __init__(self, rank: int, world_size: int, d_model=256, n_layers=4):
        super().__init__()
        self.rank = rank
        self.world_size = world_size

        self.embed = nn.Embedding(10000, d_model)
        self.layers = nn.ModuleList([
            Zero3TransformerLayer(d_model, rank, world_size)
            for _ in range(n_layers)
        ])
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 10000)

    def forward(self, input_ids):
        x = self.embed(input_ids)

        # ZeRO-3 逐层计算
        for i, layer in enumerate(self.layers):
            # 每层: All-Gather → 计算 → 释放
            x = layer(x)

        # 最后 All-Gather 输出层参数
        x = self.ln(x)
        x = self.head(x)
        return x


def demo():
    """演示 ZeRO-3 分片和动态物化"""
    print("=" * 60)
    print("ZeRO-3: 参数分片 + 动态物化 Demo")
    print("=" * 60)

    world_size = 4

    # 创建 4 个 rank 的模型实例
    models = []
    for rank in range(world_size):
        m = ManualZeRO3Model(rank, world_size, d_model=64, n_layers=2)
        models.append(m)

    total_params = sum(p.numel() for p in models[0].parameters())
    local_params = sum(p.local_numel for layer in models[0].layers
                       for p in layer.sharded_params)
    # 加上 embedding, ln, head 的分片
    # (简化: 只演示概念)

    print()
    print("显存对比 (模拟 Ψ=65B, N=64):")
    print(f"  DDP 每卡:      16Ψ = {16*65:.0f} GB")
    print(f"  ZeRO-3 每卡:  16Ψ/N = {16*65/64:.1f} GB   (-{100-100/64:.0f}%)")
    print()
    print("通信量对比 (每 step):")
    print(f"  DDP:    All-Reduce    2Ψ")
    print(f"  ZeRO-3: All-Gather×2 + Reduce-Scatter = 3Ψ")
    print(f"          前向: All-Gather(Ψ) + 反向: All-Gather(Ψ) + Reduce-Scatter(Ψ)")
    print()
    print("----- ZeRO-3 前向流程 -----")
    print("Layer 0: [All-Gather W₀ ████████████████] [计算 F₀ ████████████████████] [释放 W₀]")
    print("Layer 1:                     [All-Gather W₁ ██████████] [计算 F₁ ██████]")
    print("                               ↑ 与 F₀ 重叠!")
    print()
    print("关键: 通信与计算重叠 → 额外通信开销被掩盖")

if __name__ == "__main__":
    demo()
```

```bash
python zero_from_scratch/part3_zero3.py
```

---

## Part 4: 迷你 DeepSpeed — 完整整合

**目标**: 将 Part 1-3 整合成一个可运行的迷你 DeepSpeed。

创建 `part4_mini_deepspeed.py`:

```python
"""
迷你 DeepSpeed: 从零实现 ZeRO-3 训练框架

包含:
  1. 参数分片 (Parameter Sharding)
  2. 动态物化 (Dynamic Materialization)
  3. 逐层 All-Gather/Reduce-Scatter
  4. 分片 Adam 更新
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

class ParameterShard:
    """单个参数的 ZeRO-3 分片"""
    def __init__(self, param: nn.Parameter, rank: int, world_size: int):
        self.rank = rank
        self.world_size = world_size
        self.full_shape = param.shape
        self.full_numel = param.numel()

        # 计算分片边界
        chunk_size = full_numel // world_size
        remainder = full_numel % world_size
        self.start = rank * chunk_size + min(rank, remainder)
        self.end = self.start + chunk_size + (1 if rank < remainder else 0)

        # 本地只存 1/N
        flat = param.data.reshape(-1)
        self.local_data = flat[self.start:self.end].clone()
        self.local_grad = torch.zeros_like(self.local_data)

        # Adam 状态 (本地分片)
        self.momentum = torch.zeros_like(self.local_data)
        self.variance = torch.zeros_like(self.local_data)

    def gather(self) -> torch.Tensor:
        """All-Gather: 收集完整参数 (模拟)"""
        full = torch.zeros(self.full_numel)
        # 本 rank 写入自己的分片
        full[self.start:self.end] = self.local_data
        return full.reshape(self.full_shape)

    def scatter_grad(self, full_grad: torch.Tensor):
        """Reduce-Scatter: 只保留本 rank 的梯度分片"""
        flat = full_grad.reshape(-1)
        self.local_grad = flat[self.start:self.end].clone()

    def update(self, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        """Adam 更新 (只更新本地分片)"""
        self.momentum.mul_(beta1).add_(self.local_grad, alpha=1-beta1)
        self.variance.mul_(beta2).add_(self.local_grad**2, alpha=1-beta2)
        m_hat = self.momentum / (1-beta1)
        v_hat = self.variance / (1-beta2)
        self.local_data.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)


class MiniDeepSpeedEngine:
    """
    迷你 DeepSpeed 引擎

    模拟 ZeRO-3 的核心行为:
      每层前向: All-Gather 参数 → 计算 → 释放
      每层反向: All-Gather 参数 → 算梯度 → Reduce-Scatter → 释放
      更新: 每卡只更新本地分片
    """
    def __init__(self, model: nn.Module, rank: int, world_size: int):
        self.model = model
        self.rank = rank
        self.world_size = world_size
        self.shards: Dict[str, ParameterShard] = {}

        # 为每个参数创建分片
        for name, param in model.named_parameters():
            self.shards[name] = ParameterShard(param, rank, world_size)

        print(f"[MiniDS Rank {rank}] 分片完成: {len(self.shards)} 个参数, "
              f"每卡 {sum(s.local_data.numel() for s in self.shards.values()):,} 参数本地")

    def forward(self, x):
        """
        ZeRO-3 前向:
          for each layer:
            All-Gather 该层参数
            计算
            释放非本地分片
        """
        # 简化: 按 module 逐层处理
        for name, module in self.model.named_children():
            layer_shards = {n: s for n, s in self.shards.items()
                          if n.startswith(name)}

            # All-Gather (模拟)
            for n, s in layer_shards.items():
                full_param = s.gather()

            # 把完整参数临时注入 module
            x = module(x)

            # 释放临时参数 (只保留分片)
            # (此处 PyTorch 自动管理)

        return x

    def backward(self, loss):
        loss.backward()
        # 梯度已经计算好, 在 param.grad 中

    def step(self, lr=1e-3):
        """分片 Adam 更新"""
        for name, shard in self.shards.items():
            # 从 param.grad 提取对应分片部分
            full_grad = self.get_param_grad(name)
            if full_grad is not None:
                shard.scatter_grad(full_grad)

            # 只更新本地分片
            shard.update(lr=lr)

            # 写回 model 的对应分片
            self.set_param_data(name, shard.local_data)

    def get_param_grad(self, name):
        """获取参数梯度"""
        param = dict(self.model.named_parameters())[name]
        return param.grad

    def set_param_data(self, name, data):
        """写回参数数据到 model"""
        param = dict(self.model.named_parameters())[name]
        flat = param.data.reshape(-1)
        shard = self.shards[name]
        flat[shard.start:shard.end] = data.to(param.dtype)


def demo():
    """演示迷你 DeepSpeed 训练循环"""
    print("=" * 60)
    print("迷你 DeepSpeed — 从零实现 ZeRO-3")
    print("=" * 60)

    model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )

    # 模拟 4 个 rank
    world_size = 4
    engines = []
    for rank in range(world_size):
        engine = MiniDeepSpeedEngine(model, rank, world_size)
        engines.append(engine)

    print()
    print("训练 5 steps...")
    for step in range(5):
        x = torch.randn(8, 64)
        y = torch.randint(0, 10, (8,))

        for rank, engine in enumerate(engines):
            # 前向 (All-Gather 逐层)
            logits = engine.forward(x)

            # Loss
            loss = nn.CrossEntropyLoss()(logits, y)
            print(f"  Step {step}, Rank {rank}: loss={loss.item():.4f}")

            # 反向
            engine.backward(loss)

            # 分片更新
            engine.step(lr=0.01)

    print()
    print("关键理解:")
    print("  1. 每卡只存 1/N 参数 + 1/N 梯度 + 1/N 优化器状态")
    print("     = 显存从 16Ψ 降至 16Ψ/N")
    print("  2. 计算时才 All-Gather 完整参数, 算完释放")
    print("  3. 更新不需要通信 (每卡独立更新本地分片)")
    print("  4. 代价: 前向/反向各多一次 All-Gather")

if __name__ == "__main__":
    demo()
```

```bash
python zero_from_scratch/part4_mini_deepspeed.py
```

---

## Part 5: 用多进程模拟多卡训练

**目标**: 用 `torch.multiprocessing` 模拟 4 个 GPU rank, 真正看到分片行为。

创建 `part5_multiprocess.py`:

```python
"""
多进程模拟多卡: 真实看到 ZeRO-3 的分片

用 torch.multiprocessing 启动多个进程,
每个进程模拟一个 GPU rank, 打印各自的分片内容。
"""

import torch
import torch.nn as nn
import torch.multiprocessing as mp

def train_rank(rank: int, world_size: int, results: dict):
    """单个 rank 的训练进程"""
    # 每个 rank 创建一个模型实例 (但参数各自分片)
    model = nn.Sequential(
        nn.Linear(16, 32),
        nn.ReLU(),
        nn.Linear(32, 8),
    )

    # 打印各 rank 的参数分片情况
    total_params = sum(p.numel() for p in model.parameters())
    params_per_rank = total_params // world_size
    remainder = total_params % world_size

    start = rank * params_per_rank + min(rank, remainder)
    end = start + params_per_rank + (1 if rank < remainder else 0)

    print(f"[Rank {rank}] 总参数={total_params}, "
          f"本地分片=[{start}:{end}] ({end-start} 个参数, "
          f"占 {100*(end-start)/total_params:.1f}%)")

    # 存储结果用于验证
    results[rank] = (start, end, total_params)

def demo():
    """启动 4 个进程模拟 4 卡"""
    world_size = 4
    manager = mp.Manager()
    results = manager.dict()

    processes = []
    for rank in range(world_size):
        p = mp.Process(target=train_rank, args=(rank, world_size, results))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # 验证: 所有分片不重叠且覆盖全部参数
    total = 0
    intervals = []
    for rank in range(world_size):
        start, end, total_params = results[rank]
        intervals.append((start, end))
        print(f"  Rank {rank}: [{start}, {end})")

    # 检查是否完整覆盖
    merged = sorted(intervals)
    assert merged[0][0] == 0
    assert merged[-1][1] == total_params
    for i in range(len(merged)-1):
        assert merged[i][1] == merged[i+1][0], "参数分片不连续!"
    print(f"  ✅ 分片完整覆盖 {total_params} 个参数, 无重叠")

if __name__ == "__main__":
    # 需要 spawn 模式以支持 CUDA
    mp.set_start_method("spawn", force=True)
    demo()
```

```bash
python zero_from_scratch/part5_multiprocess.py
```

---

## 总结: 你复现了什么

```
Part 1: All-Reduce vs Reduce-Scatter 通信量差异       (ZeRO-2 核心)
Part 2: Adam 优化器状态分片, 每卡只维护 1/N           (ZeRO-1 核心)
Part 3: 参数分片 + 逐层 All-Gather + 动态释放          (ZeRO-3 核心)
Part 4: 整合为迷你 DeepSpeed 训练引擎                  (完整框架)
Part 5: 多进程模拟多卡, 验证分片正确性                  (分布式验证)
```

对应到 DeepSpeed 源码位置:
- `deepspeed/runtime/zero/stage1.py` — ZeRO-1
- `deepspeed/runtime/zero/stage2.py` — ZeRO-2
- `deepspeed/runtime/zero/stage3.py` — ZeRO-3

理解这些, 你就掌握了 DeepSpeed 的核心。
