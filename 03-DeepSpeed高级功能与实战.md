# 03 - DeepSpeed 高级功能与实战

## 一、DeepSpeed 架构概览

```
                    ┌─────────────────────────┐
                    │    User Training Script  │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   DeepSpeed Engine       │
                    └───────────┬─────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          ▼                     ▼                     ▼
   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
   │ZeRO Optimizer│    │ Parallelism  │    │   Runtime    │
   │  ZeRO-1/2/3  │    │  TP/PP/DP    │    │  Comm Mgmt   │
   │  Offload     │    │  Sequence Par│    │  Checkpoint  │
   └──────────────┘    └──────────────┘    └──────────────┘
          │                    │                    │
          ▼                    ▼                    ▼
   ┌─────────────────────────────────────────────────────┐
   │                  NCCL / Backend                      │
   └─────────────────────────────────────────────────────┘
```

---

## 二、配置文件详解

DeepSpeed 通过 JSON 配置文件集中管理所有训练参数。

### 2.1 最小配置

```json
{
  "train_batch_size": 32,
  "gradient_accumulation_steps": 1,
  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 3e-5,
      "betas": [0.9, 0.999],
      "eps": 1e-8,
      "weight_decay": 0.01
    }
  },
  "scheduler": {
    "type": "WarmupLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 3e-5,
      "warmup_num_steps": 1000
    }
  },
  "zero_optimization": {
    "stage": 2
  },
  "fp16": {
    "enabled": true
  }
}
```

### 2.2 完整生产配置

```json
{
  "train_batch_size": 256,
  "train_micro_batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 8,

  "optimizer": {
    "type": "AdamW",
    "params": {
      "lr": 1e-4,
      "betas": [0.9, 0.95],
      "eps": 1e-8,
      "weight_decay": 0.1
    }
  },

  "scheduler": {
    "type": "WarmupDecayLR",
    "params": {
      "warmup_min_lr": 0,
      "warmup_max_lr": 1e-4,
      "warmup_num_steps": 2000,
      "total_num_steps": 100000
    }
  },

  "zero_optimization": {
    "stage": 3,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "stage3_prefetch_bucket_size": 5e8,
    "stage3_param_persistence_threshold": 1e6,
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e8,
    "stage3_gather_16bit_weights_on_model_save": true
  },

  "fp16": {
    "enabled": true,
    "auto_cast": true,
    "loss_scale": 0,
    "initial_scale_power": 16,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  },

  "bf16": {
    "enabled": false
  },

  "activation_checkpointing": {
    "partition_activations": true,
    "cpu_checkpointing": true,
    "number_checkpoints": null,
    "synchronize_checkpoint_boundary": false,
    "profile": false
  },

  "communication_data_type": "bf16",
  "gradient_clipping": 1.0,
  "prescale_gradients": false,
  "wall_clock_breakdown": false,

  "tensorboard": {
    "enabled": true,
    "output_path": "./logs/",
    "job_name": "llama_training"
  },

  "flops_profiler": {
    "enabled": false,
    "profile_step": 10,
    "module_depth": -1
  }
}
```

### 2.3 关键参数解读

| 参数 | 作用 | 建议 |
|------|------|------|
| `overlap_comm` | 通信与计算重叠 | ZeRO-2/3 务必开启 |
| `contiguous_gradients` | 梯度存为连续缓冲区 | 开启以提高通信效率 |
| `allgather_bucket_size` | All-Gather 桶大小 | 大模型设 5e8~1e9 |
| `stage3_prefetch_bucket_size` | 预取参数桶大小 | 设为 allgather_bucket_size 的 1/2 |
| `stage3_max_live_parameters` | 最多保留的参数数 | 越大通信越少，显存越多 |
| `stage3_gather_16bit_weights_on_model_save` | 保存时收集完整参数 | 保存 checkpoint 时必开 |

---

## 三、ZeRO-3 参数调优指南

### 3.1 通信与内存的权衡

```
减少通信                   增加通信
◄──────────────────────────►
  更少 All-Gather            更多 All-Gather
  更多显存占用               更少显存占用

调优参数:
  stage3_max_live_parameters: ↑ = 更多显存, 更少通信
  stage3_prefetch_bucket_size: ↑ = 更多显存, 更好重叠
  stage3_max_reuse_distance: ↑ = 更多显存, 更好复用
```

### 3.2 调优步骤

1. **基准测试**: 先用 ZeRO-2 跑小规模，确定基线吞吐
2. **启动 ZeRO-3**: 所有参数用默认值 (~1e9)
3. **降低显存**: 如果 OOM，降低 bucket size 到 1e8~5e8
4. **优化性能**: 逐步增大 `stage3_max_live_parameters`，观察吞吐
5. **预取优化**: 调整 `stage3_prefetch_bucket_size` 为 `allgather_bucket_size` 的 30-50%

### 3.3 常见问题诊断

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| OOM | 显存不足 | 降低 batch size，减少 max_live_parameters |
| 通信慢 | IB 未启用 | 检查 NCCL 网络配置 |
| 吞吐低 | 重叠不好 | 增大 prefetch_bucket_size |
| 训练不稳定 | Loss scale 问题 | 检查 fp16/bf16 配置 |
| 卡住 | 通信死锁 | 检查 NCCL timeout, 增加 async grad allreduce |

---

## 四、集成到训练脚本

### 4.1 基础集成

```python
import deepspeed
import torch
from transformers import AutoModel, AutoTokenizer

# 1. 定义模型
model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. 定义数据加载器
train_dataset = ...
train_loader = DataLoader(train_dataset, batch_size=4)

# 3. 初始化 DeepSpeed Engine
model_engine, optimizer, train_loader, _ = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    training_data=train_dataset,
    config_params="ds_config.json"
)

# 4. 训练循环
for epoch in range(num_epochs):
    for batch in train_loader:
        outputs = model_engine(**batch)
        loss = outputs.loss
        
        model_engine.backward(loss)     # 替代 loss.backward()
        model_engine.step()             # 替代 optimizer.step()
```

### 4.2 使用 HuggingFace Trainer 集成

```python
from transformers import Trainer, TrainingArguments
import deepspeed

training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=3e-5,
    num_train_epochs=3,
    fp16=True,
    deepspeed="ds_config.json",  # 直接传入配置文件
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
```

### 4.3 自定义 ZeRO-3 适配

某些模型需要特殊处理 ZeRO-3 的参数收集:

```python
import deepspeed
from deepspeed.zero import GatheredParameters

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(32000, 4096)
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(32)])
        self.lm_head = nn.Linear(4096, 32000, bias=False)
        self.embedding.weight = self.lm_head.weight  # weight tying

    def forward(self, input_ids):
        # ZeRO-3 下需要对 tied weights 做特殊处理
        with GatheredParameters([self.embedding.weight]):
            x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        with GatheredParameters([self.lm_head.weight]):
            logits = self.lm_head(x)
        return logits
```

---

## 五、Activation Checkpointing

### 5.1 DeepSpeed 实现

```python
import deepspeed

# 方法 1: 通过配置启用
# "activation_checkpointing": {
#     "partition_activations": true,
#     "cpu_checkpointing": true
# }

# 方法 2: 代码中手动指定 checkpoint 点
from deepspeed.runtime.activation_checkpointing import checkpointing

# 替换模型中的关键层
def forward(self, x):
    # 对这个 Transformer 块启用 checkpointing
    x = checkpointing.checkpoint(self.transformer_block, x)
    return x
```

### 5.2 与 HuggingFace 结合

```python
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from deepspeed.runtime.activation_checkpointing import checkpointing

# 设置 checkpointing
checkpointing.configure(
    partition_activations=True,
    cpu_checkpointing=True,
    contiguous_memory_optimization=True,
    profile=True,
)

# 用 checkpoint 包裹 decoder layers
for i, layer in enumerate(model.model.layers):
    model.model.layers[i] = checkpointing.checkpoint(layer)
```

---

## 六、ZeRO-3 权重保存与加载

### 6.1 保存 (需要收集完整权重)

```python
# 方法 1: DeepSpeed 引擎保存 (ZeRO-3 自动收集)
model_engine.save_checkpoint(
    save_dir="./checkpoint",
    client_state={
        "epoch": epoch,
        "best_val_loss": best_loss,
    }
)

# 方法 2: 合并保存为单文件 (ZeRO-3 需要 stage3_gather_16bit_weights_on_model_save=true)
if model_engine.zero_optimization_stage() == 3:
    # 收集完整模型权重 (All ranks 参与)
    state_dict = model_engine._zero3_consolidated_16bit_state_dict()
    if torch.distributed.get_rank() == 0:
        torch.save(state_dict, "pytorch_model.bin")
```

### 6.2 加载

```python
# 从 DeepSpeed checkpoint 恢复
_, client_state = model_engine.load_checkpoint(
    load_dir="./checkpoint",
    load_optimizer_states=True,
    load_lr_scheduler_states=True,
)

# 从合并的 checkpoint 加载
state_dict = torch.load("pytorch_model.bin")
model_engine.load_state_dict(state_dict, strict=False)
```

### 6.3 分布式权重转换

```python
# ZeRO-3 分片权重 → 单文件 (可用 convert_zero_checkpoint.py)
!python -m deepspeed.utils.convert_zero_checkpoint \
    --input_dir ./zero_checkpoint/global_step1000 \
    --output_dir ./converted_checkpoint
```

---

## 七、Flops Profiler 与性能分析

### 7.1 使用 Flops Profiler

```python
with deepspeed.profiling.flops_profiler.profile(
    model_engine,
    next(iter(train_loader)),
    warmup_steps=5,
    profile_step=10,
) as prof:
    # 运行推理/训练
    prof.print_model_profile()
    prof.start_profile()
    for _ in range(10):
        model_engine(**batch)
    prof.stop_profile()
    prof.print_profile_output()
```

### 7.2 输出解读

```
-------------------------- Flops Profiler -------------------
Model FLOPs: 1.95e15 (1.95 PFLOPS)
Model MFU: 52.3%
Runtime per step: 2.35s
Sample throughput: 108.9 samples/sec
Memory: 62.3 GB / 80 GB (77.9%)

Top 5 ops by compute time:
  torch._C._nn.linear: 42.1%
  torch._C._nn.attention: 28.3%
  torch._C._nn.layer_norm: 5.2%
  torch._C._nn.embedding: 3.1%
  torch._C._nn.dropout: 1.2%

Communication overhead:
  All-Gather: 8.3% of step time
  Reduce-Scatter: 5.1% of step time
----------------------------------------------------------
```

---

## 八、训练启动命令

### 8.1 单节点多卡

```bash
# 单节点 8 卡
deepspeed --num_gpus=8 train.py \
    --deepspeed_config ds_config.json \
    --model_name meta-llama/Llama-2-7b-hf

# 或使用 torchrun
torchrun --nproc_per_node=8 train.py \
    --deepspeed_config ds_config.json
```

### 8.2 多节点

```bash
# 方法 1: deepspeed 启动器 (推荐)
deepspeed --num_gpus=8 \
    --num_nodes=16 \
    --master_addr=node0 \
    --master_port=29500 \
    --hostfile=hostfile.txt \
    train.py --deepspeed_config ds_config.json

# hostfile.txt 示例:
# node0 slots=8
# node1 slots=8
# node2 slots=8
# ...

# 方法 2: torchrun 多节点
torchrun --nnodes=16 \
    --nproc_per_node=8 \
    --rdzv_endpoint=node0:29500 \
    --rdzv_backend=c10d \
    train.py --deepspeed_config ds_config.json
```

### 8.3 Slurm 集群

```bash
#!/bin/bash
#SBATCH --job-name=llama_training
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=4

# 设置 NCCL 环境
export NCCL_DEBUG=INFO
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=4
export NCCL_IB_HCA=mlx5_0,mlx5_1

deepspeed \
    --num_gpus=8 \
    --num_nodes=$SLURM_NNODES \
    --master_addr=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -1) \
    --master_port=29500 \
    train.py \
    --deepspeed_config ds_config.json
```

---

## 九、常见问题排查

### 9.1 NCCL 通信问题

```bash
# 启用 NCCL 调试
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

# 使用特定网卡
export NCCL_IB_HCA=mlx5_0,mlx5_1

# 禁用 IB (回退到 TCP, 用于排查)
export NCCL_IB_DISABLE=1

# 设置超时
export NCCL_TIMEOUT=1800
```

### 9.2 显存问题

```bash
# 查看实际显存分配
nvidia-smi -l 1

# PyTorch 显存分析
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512

# DeepSpeed 显存日志
export DS_ACCELERATOR_CUDA_ENABLE_MEMORY_STATS=1
```

### 9.3 训练崩溃

- **CUDA OOM**: 减小 batch size, 开启 activation checkpointing, 降低 ZeRO-3 bucket size
- **NaN loss**: 检查学习率, 启用 gradient clipping, 检查数据是否有坏样本
- **hang/stuck**: 检查 NCCL timeout, 确认 IB/RoCE 配置正确, 检查显卡间 P2P

---

## 思考题

1. ZeRO-3 的 `stage3_max_live_parameters` 设为 0 会怎样？设为无穷大呢？
2. 为什么 HuggingFace Trainer 集成 DeepSpeed 比手动初始化更容易 OOM？
3. 在什么场景下应该使用 `bf16` 而非 `fp16`？
