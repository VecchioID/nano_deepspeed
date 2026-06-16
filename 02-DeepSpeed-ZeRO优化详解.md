# 02 - DeepSpeed ZeRO 优化详解

## 一、ZeRO 的动机

### 1.1 数据并行中的冗余

在标准 DDP 中，每张卡存储完整的：
- **模型参数** (Parameters)
- **梯度** (Gradients)
- **优化器状态** (Optimizer States: momentum, variance)

对于 Adam 优化器，显存占用分布：

```
┌──────────────────────────────────────┐
│          每个 GPU 的显存              │
├──────────────────────────────────────┤
│  参数:   Ψ × 2 bytes        (FP16)   │
│  梯度:   Ψ × 2 bytes        (FP16)   │
│  优化器:  Ψ × 4 × 2 bytes   (FP32)   │  ← momentum + variance + FP32参数
│  Activation: 取决于 batch size        │
└──────────────────────────────────────┘
```

**关键洞察**: 在 N 卡 DDP 中，模型参数和优化器状态被复制了 N 份。如果能消除冗余，单卡显存占用可降低 N 倍。

### 1.2 ZeRO 的核心思想

**ZeRO = Zero Redundancy Optimizer**

将模型状态（参数、梯度、优化器状态）**分片**到所有数据并行进程中，每个进程只持有 1/N，在需要时通过通信集体获取。

```
Naive DDP:                     ZeRO-3:
┌──────┐┌──────┐┌──────┐     ┌──────┐┌──────┐┌──────┐
│ P₀~₂ ││ P₀~₂ ││ P₀~₂ │     │  P₀  ││  P₁  ││  P₂  │
│ G₀~₂ ││ G₀~₂ ││ G₀~₂ │     │  G₀  ││  G₁  ││  G₂  │
│ O₀~₂ ││ O₀~₂ ││ O₀~₂ │     │  O₀  ││  O₁  ││  O₂  │
└──────┘└──────┘└──────┘     └──────┘└──────┘└──────┘
  复制3份                      各持1/3, 无冗余
```

---

## 二、ZeRO 三阶段 (ZeRO-1, ZeRO-2, ZeRO-3)

### 2.1 ZeRO-1: 优化器状态分片

**分片内容**: 仅分片优化器状态 (Adam momentum + variance + FP32 主权重)

**显存收益**:
```
减少量 = (Ψ × 4 × 2 bytes) × (N-1)/N
总占用 = Ψ×2 (param) + Ψ×2 (grad) + Ψ×4×2/N (optimizer)
```

**通信**:
- 前向/反向: **零额外通信**
- 更新阶段: 各进程持有部分优化器状态，只更新自己分到的参数
- 完成后需要 All-Gather 同步完整参数（仅在更新后）

**局限**: 梯度仍然是完整的（未分片）

### 2.2 ZeRO-2: 梯度分片

**分片内容**: 优化器状态 + 梯度

**梯度分片机制**:
1. 反向传播完成后，每个进程持有完整的梯度
2. 使用 Reduce-Scatter 替代 All-Reduce:
   - All-Reduce: 求和 + 广播（通信量 2Ψ）
   - Reduce-Scatter: 求和 + 分发（通信量 Ψ）
3. 每个进程只保留自己分片部分的梯度

**显存收益**:
```
总占用 = Ψ×2 (param) + Ψ×2/N (grad) + Ψ×4×2/N (optimizer)
```

**通信**:
- Reduce-Scatter 比 All-Reduce 少一半通信量
- 通信与计算可重叠（逐层 Reduce-Scatter）

### 2.3 ZeRO-3: 参数分片

**分片内容**: 优化器状态 + 梯度 + 模型参数

**参数分片的核心机制**:

```
前向传播:
  Layer i 开始 → 从所有进程 All-Gather 参数 → 计算 Layer i
              → 丢弃其他进程的参数（仅保留自己分片） → 转入 Layer i+1

反向传播:
  Layer i 开始 → 从所有进程 All-Gather 参数 → 计算梯度
              → 丢弃其他进程的参数 → 继续反向
```

**动态参数物化 (Dynamic Parameter Materialization)**:
- 计算某层时才收集该层参数
- 计算完成后立即释放（除了自己分片的部分）
- 时间换空间: 增加通信开销，但极大降低显存

**显存收益**:
```
总占用 = Ψ×2/N (param) + Ψ×2/N (grad) + Ψ×4×2/N (optimizer)
```

**通信开销**:
```
DDP 通信量:  2Ψ (每个 batch)
ZeRO-3 通信量: 3Ψ (前向 All-Gather Ψ + 反向 All-Gather Ψ + Reduce-Scatter Ψ)
```

### 2.4 三阶段对比

| 维度 | ZeRO-1 | ZeRO-2 | ZeRO-3 |
|------|--------|--------|--------|
| 分片内容 | Optimizer | Optimizer + Grad | Optimizer + Grad + Param |
| 显存节省 | 4× | 8× | 近 N× |
| 额外通信 | 无 | 无 (更优) | 2 次 All-Gather |
| 计算-通信重叠 | 不易 | 梯度阶段可重叠 | 前向/反向均可重叠 |
| 适用场景 | 小规模扩展 | 中等规模 | 超大规模 |

**显存节省倍数 (N=64 卡, Adam)**:

```
Naive:  16 + 4 + 4 = 24 单位 (param FP32/FP16 + grad + optim)
ZeRO-1: 16 + 4 + 4/64 ≈ 20  (节省 16%)
ZeRO-2: 16 + 4/64 + 4/64 ≈ 16 (节省 33%)
ZeRO-3: 16/64 + 4/64 + 4/64 ≈ 0.375 (节省 98.4%)
```

---

## 三、ZeRO 的通信优化

### 3.1 计算-通信重叠

ZeRO 最核心的性能优化是将通信隐藏在计算背后。

**ZeRO-3 重叠示意**:

```
时间 →
前向:
  Layer 0: [All-Gather W₀ ██████] [计算 F₀ ████████████████] [释放 W₀]
  Layer 1:                     [All-Gather W₁ ██████] [计算 F₁ ██████]
  (All-Gather 与计算 Layer i 可重叠，同时进行)

反向:
  Layer N: [计算 B_N ████████████████] [Reduce-Scatter G_N ██████]
  Layer N-1:                           [计算 B_{N-1} ████████████████]
```

**实现细节**:
- 使用异步通信 (NCCL `ncclGroupStart/End`, CUDA events)
- 预取机制: 在计算当前层时预取下一层的参数
- 通信 stream 与计算 stream 并行执行

### 3.2 Bucket 合并

- 将多个小张量的通信合并为一个大张量的通信
- 提高带宽利用率（小消息延迟大，带宽利用率低）
- 默认 bucket size: 5e8 (500M) 参数

### 3.3 梯度累积阶段的分片

启用 Gradient Accumulation 时:
- ZeRO-2/3 在梯度累积期间不需要额外通信
- 每个 micro-step 本地计算梯度，只在最终 step 做 Reduce-Scatter

---

## 四、ZeRO-Offload 与 ZeRO-Infinity

### 4.1 ZeRO-Offload

**动机**: GPU 显存仍然不够时，利用 CPU 内存（DRAM）作为额外存储层。

**原理**:
```
┌──────────────────────────────────────────────────┐
│                    GPU                           │
│  ┌──────────┐  ┌──────────┐  ┌────────────────┐ │
│  │ 参数分片  │  │ 梯度分片  │  │ 计算 + Activation│ │
│  └────┬─────┘  └────┬─────┘  └────────────────┘ │
└───────┼──────────────┼──────────────────────────┘
        │  offload     │  offload
        ▼              ▼
┌──────────────────────────────────────────────────┐
│                   CPU/NUMA                       │
│  ┌──────────────────────────────────────────────┐│
│  │      优化器状态 (FP32 momentum + variance)    ││
│  └──────────────────────────────────────────────┘│
└──────────────────────────────────────────────────┘
```

**通信路径**:
- 参数/梯度 offload: GPU → CPU (PCIe: ~64 GB/s)
- 参数/梯度回传: CPU → GPU (PCIe: ~64 GB/s)
- 与 NVLink (900 GB/s) 相比慢了 10×+
- **因此仅 offload optimizer states, 参数和梯度保留在 GPU**

**性能影响**:
- Offload optimizer: 通常 10-20% 性能损失
- Offload parameters: 30-50% 性能损失（不推荐）

### 4.2 ZeRO-Infinity

**进一步扩展**: 利用 NVMe SSD 作为第三级存储。

```
存储层级:  GPU HBM → CPU DRAM → NVMe SSD
延迟:       ~1μs    → ~0.1μs  → ~10μs (读)
带宽:     ~2000GB/s → ~100GB/s → ~7GB/s (PCIe 4.0 ×4)
```

**特点**:
- 理论上可训练无限大的模型
- 实际受限于 NVMe 带宽，通常仅作为兜底
- 自动根据访问频率决定 offload 层级

---

## 五、ZeRO 与张量并行 (TP) 的关系

### 5.1 本质区别

| 维度 | ZeRO-3 (DP 分片) | Tensor Parallelism |
|------|-------------------|--------------------|
| 通信粒度 | 逐层 (Per Layer) | 逐算子 (Per Op) |
| 通信量 | O(Ψ) 每层 | O(h) 每层 (h=hidden) |
| 通信延迟 | 少量大消息 | 大量小消息 |
| 扩展上限 | 千卡级 | 通常 ≤ 8 卡 |
| 跨节点 | 友好 | 困难 |

### 5.2 组合使用

实际大规模训练通常: **TP + ZeRO-1/2 + PP**

```
TP 组内: 张量并行 (解决单算子显存)
ZeRO-1/2: DP 组内分片优化器/梯度
PP: 流水线并行 (降低跨节点通信)
```

**为什么不 TP + ZeRO-3？**
- ZeRO-3 的逐层 All-Gather 与 TP 的逐算子通信冲突
- 两者都要求频繁 All-Gather，叠加后通信开销剧增
- 实践中 ZeRO-3 通常作为 TP 的替代方案而非补充

---

## 六、ZeRO 训练流程伪代码

```
ZeRO-3 训练 loop:

for each batch:
    # 前向传播
    for layer in model:
        # All-Gather: 收集当前层完整参数
        nccl.all_gather(layer.params, from_all_ranks)
        
        # 计算前向
        output = layer.forward(input)
        
        # 释放非本 rank 的参数分片
        layer.params.free_except(my_rank_slice)
    
    # 计算 loss
    loss = criterion(output, target)
    
    # 反向传播
    for layer in reversed(model):
        # All-Gather: 收集当前层完整参数
        nccl.all_gather(layer.params, from_all_ranks)
        
        # 计算梯度
        layer.backward(grad_output)
        
        # Reduce-Scatter: 梯度求和并分发
        nccl.reduce_scatter(layer.grad, to_all_ranks)
        
        # 释放非本 rank 的参数和梯度
        layer.params.free_except(my_rank_slice)
        layer.grad.free_except(my_rank_slice)
    
    # 优化器更新 (no communication needed)
    optimizer.step(my_rank_slice_params, my_rank_slice_grads)
```

---

## 思考题

1. ZeRO-3 的 All-Gather 在通信量上比 DDP 的 All-Reduce 还多 50%，为什么实际效果仍然很好？
2. 为什么 ZeRO-3 和 TP 通常不一起使用？如果必须一起用，应该怎么做？
3. ZeRO-Offload 中，为什么选择 offload optimizer states 而不是 gradients？
